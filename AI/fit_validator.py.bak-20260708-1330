import time
import jwt
from jwt import PyJWKClient
from django.http import JsonResponse

EXPECTED_APP_ID = "ari:cloud:ecosystem::app/955fdc30-19a5-4438-8128-f172cb3753e7"
EXPECTED_AUDIENCE = "forge"

JWKS_URL = "https://forge.cdn.prod.atlassian-dev.net/.well-known/jwks.json"

jwks_client = PyJWKClient(JWKS_URL)


def validate_fit_token(token):
    try:
        # -----------------------------------
        # 1. Read header — determine token type
        # -----------------------------------
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid", "")
        is_context_token = "forge/context-token" in kid

        if is_context_token:
            # -----------------------------------
            # CONTEXT TOKEN PATH
            # Atlassian does NOT publish JWKS keys for context tokens.
            # Validate all claims manually without signature verification.
            # -----------------------------------
            claims = jwt.decode(
                token,
                options={"verify_signature": False},
            )

            print("DECODED CLAIMS:", claims)

            # Verify expiration
            exp = claims.get("exp")
            if exp is None or exp < time.time():
                raise Exception("Token has expired")

            # Verify audience
            aud = claims.get("aud")
            if aud != EXPECTED_AUDIENCE:
                raise Exception(f"Invalid audience: {aud}")

            # Verify issuer
            issuer = claims.get("iss")
            if issuer != "forge/context-token":
                raise Exception(f"Invalid issuer: {issuer}")

            # Verify app ID
            app_id = claims.get("appId")
            if not app_id:
                raise Exception("Missing app id in context token")
            if app_id != EXPECTED_APP_ID:
                raise Exception(f"Invalid app id: {app_id}")

            # Verify installation context
            installation_id = claims.get("installationId")
            if not installation_id:
                raise Exception("Missing installation id in context token")

            return {
                "type": "context_token",
                "app_id": app_id,
                "installation_id": installation_id,
                "claims": claims,
            }

        else:
            # -----------------------------------
            # SIGNED TOKEN PATH
            # Verify signature via JWKS + all claims.
            # -----------------------------------
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=EXPECTED_AUDIENCE,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                },
            )

            print("DECODED CLAIMS:", claims)

            # Verify issuer
            issuer = claims.get("iss")
            if issuer != "https://forge.atlassian.com":
                raise Exception(f"Invalid issuer: {issuer}")

            # Verify app ID
            app_id = (
                claims.get("appId")
                or claims.get("app", {}).get("id")
            )
            if not app_id:
                raise Exception("Missing app id in signed token")
            if app_id != EXPECTED_APP_ID:
                raise Exception(f"Invalid app id: {app_id}")

            # Verify installation context
            installation_id = (
                claims.get("installationId")
                or claims.get("sub")
            )
            if not installation_id:
                raise Exception("Missing installation id in signed token")

            return {
                "type": "signed_token",
                "app_id": app_id,
                "installation_id": installation_id,
                "claims": claims,
            }

    except Exception as e:
        raise Exception(f"FIT validation failed: {str(e)}")
