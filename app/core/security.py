import jwt
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

logger = logging.getLogger("uvicorn")
security_scheme = HTTPBearer(auto_error=False)

# JWK Client for Clerk JWT verification
jwk_client = jwt.PyJWKClient(settings.CLERK_JWKS_URL)

def verify_clerk_token(token: str) -> dict:
    """
    Decodes and verifies a Clerk session JWT token against Clerk's JWKS.
    """
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        data = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": True},
            leeway=timedelta(seconds=10)
        )
        return data
    except Exception as e:
        logger.error(f"Clerk JWT Verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Clerk Token: {str(e)}"
        )

def generate_access_token(clerk_id: str) -> str:
    """
    Generates a short-lived custom JWT access token.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": clerk_id,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def generate_refresh_token(clerk_id: str) -> str:
    """
    Generates a long-lived custom JWT refresh token.
    """
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {
        "sub": clerk_id,
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def hash_token(token: str) -> str:
    """
    Hashes a token using SHA-256 for secure storage.
    """
    return hashlib.sha256(token.encode()).hexdigest()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """
    Dependency that verifies the custom Access Token from the request header
    and retrieves the user record from the MongoDB users collection.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header"
        )
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "access":
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
        
        # Verify user in database
        from app.db.mongodb import get_user_collection
        users_coll = get_user_collection()
        user = await users_coll.find_one({"clerk_id": clerk_id})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in system"
            )
        
        if user.get("is_banned"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is banned"
            )
            
        return user
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token"
        )
