import logging
import jwt
from fastapi import APIRouter, Response, Request, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
from app.core.config import settings
from app.core.security import (
    verify_clerk_token,
    generate_access_token,
    generate_refresh_token,
    hash_token,
    get_current_user
)
from app.db.mongodb import get_user_collection, get_profile_collection

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/api/auth", tags=["auth"])

class VerifyRequest(BaseModel):
    token: str
    email: Optional[str] = None
    name: Optional[str] = None

@router.post("/verify")
async def verify_auth(payload: VerifyRequest, response: Response):
    """
    Endpoint called by the frontend after authentication with Clerk.
    Verifies the Clerk JWT token, updates/creates user and profile,
    and returns custom Access Token and sets Refresh Token in cookies.
    """
    clerk_token = payload.token
    
    # 1. Verify the Clerk JWT token
    clerk_data = verify_clerk_token(clerk_token)
    clerk_id = clerk_data.get("sub")
    
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clerk token payload does not contain user ID"
        )
    
    # Extract email and name if present in Clerk claims or request body
    email = payload.email or clerk_data.get("email") or (clerk_data.get("emails", [""])[0] if isinstance(clerk_data.get("emails"), list) else "")
    name = payload.name or clerk_data.get("name") or clerk_data.get("username") or ""
    
    users_coll = get_user_collection()
    profiles_coll = get_profile_collection()
    
    # 2. Check if user already exists
    user = await users_coll.find_one({"clerk_id": clerk_id})
    if not user:
        # Create new user
        new_user = {
            "clerk_id": clerk_id,
            "is_verified": True,
            "is_banned": False,
            "hashed_refresh_token": None
        }
        await users_coll.insert_one(new_user)
        user = new_user
    else:
        if user.get("is_banned"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is banned"
            )
    
    # Update or insert profile on every verification to keep it synchronized
    profile_data = {
        "updated_at": clerk_data.get("iat")
    }
    if email:
        profile_data["email"] = email
    if name:
        profile_data["name"] = name
        
    await profiles_coll.update_one(
        {"clerk_id": clerk_id},
        {"$set": profile_data},
        upsert=True
    )
    
    # 3. Generate new Access and Refresh tokens
    access_token = generate_access_token(clerk_id)
    refresh_token = generate_refresh_token(clerk_id)
    
    # 4. Hash and save the Refresh token in users collection
    hashed_refresh = hash_token(refresh_token)
    await users_coll.update_one(
        {"clerk_id": clerk_id},
        {"$set": {"hashed_refresh_token": hashed_refresh}}
    )
    
    # 5. Store Refresh token in Cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax", # Lax for development, can be None for cross-site
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    # Get profile info to return
    profile = await profiles_coll.find_one({"clerk_id": clerk_id})
    
    return {
        "access_token": access_token,
        "user": {
            "clerk_id": clerk_id,
            "email": profile.get("email") if profile else email,
            "name": profile.get("name") if profile else name
        }
    }

@router.post("/refresh")
async def refresh_auth(request: Request, response: Response):
    """
    Generates a new access token using the refresh token stored in cookies.
    """
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing from cookies"
        )
    
    try:
        # Decode and verify refresh token structure
        payload = jwt.decode(refresh_token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        clerk_id = payload.get("sub")
        if not clerk_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Verify in DB and match hashed refresh token
        users_coll = get_user_collection()
        user = await users_coll.find_one({"clerk_id": clerk_id})
        if not user or user.get("is_banned"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or banned"
            )
        
        stored_hash = user.get("hashed_refresh_token")
        if not stored_hash or hash_token(refresh_token) != stored_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token is invalid or has been revoked"
            )
        
        # Generate new tokens (rotate refresh token too)
        new_access = generate_access_token(clerk_id)
        new_refresh = generate_refresh_token(clerk_id)
        
        # Update DB hash
        await users_coll.update_one(
            {"clerk_id": clerk_id},
            {"$set": {"hashed_refresh_token": hash_token(new_refresh)}}
        )
        
        # Set new cookie
        response.set_cookie(
            key="refresh_token",
            value=new_refresh,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        return {"access_token": new_access}
        
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}"
        )

@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    Revokes the current refresh token and clears the cookie.
    """
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        try:
            payload = jwt.decode(refresh_token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            clerk_id = payload.get("sub")
            if clerk_id:
                users_coll = get_user_collection()
                await users_coll.update_one(
                    {"clerk_id": clerk_id},
                    {"$set": {"hashed_refresh_token": None}}
                )
        except Exception:
            pass # Ignore invalid tokens on logout
            
    response.delete_cookie(key="refresh_token")
    return {"message": "Successfully logged out"}


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    mobile: Optional[str] = None
    address: Optional[str] = None
    bio: Optional[str] = None


@router.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    clerk_id = current_user.get("clerk_id")
    profiles_coll = get_profile_collection()
    profile = await profiles_coll.find_one({"clerk_id": clerk_id})
    if not profile:
        return {
            "name": "",
            "email": "",
            "avatar_url": "",
            "mobile": "",
            "address": "",
            "bio": ""
        }
    return {
        "name": profile.get("name") or "",
        "email": profile.get("email") or "",
        "avatar_url": profile.get("avatar_url") or "",
        "mobile": profile.get("mobile") or "",
        "address": profile.get("address") or "",
        "bio": profile.get("bio") or ""
    }


@router.post("/profile")
async def update_profile(data: ProfileUpdate, current_user: dict = Depends(get_current_user)):
    clerk_id = current_user.get("clerk_id")
    profiles_coll = get_profile_collection()
    
    update_data = {}
    if data.name is not None:
        update_data["name"] = data.name
    if data.avatar_url is not None:
        update_data["avatar_url"] = data.avatar_url
    if data.mobile is not None:
        update_data["mobile"] = data.mobile
    if data.address is not None:
        update_data["address"] = data.address
    if data.bio is not None:
        update_data["bio"] = data.bio
        
    await profiles_coll.update_one(
        {"clerk_id": clerk_id},
        {"$set": update_data},
        upsert=True
    )
    return {"status": "success", "message": "Profile updated successfully"}
