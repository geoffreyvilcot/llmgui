import base64
import hashlib
from cryptography.fernet import Fernet

def generate_key_from_text(text):
    # Créer un hash SHA-256 du texte
    hash_object = hashlib.sha256(text.encode())
    # Obtenir les 32 premiers octets du hash
    key = hash_object.digest()[:32]
    # Encoder la clé en base64 URL-safe
    url_safe_key = base64.urlsafe_b64encode(key)
    return url_safe_key

def encode_text(text, passphrase):
    key = generate_key_from_text(passphrase)

    f = Fernet(key)
    token = f.encrypt(text.encode())
    return token.decode()


def decode_text(text, passphrase):
    key = generate_key_from_text(passphrase)

    f = Fernet(key)
    cleartext = f.decrypt(text).decode()
    return cleartext

if __name__ == '__main__':
    passphrase = "je me promene dans les bois"
    cifer = encode_text("A really secret message. Not for prying eyes.", passphrase)
    print(cifer)
    cleartext = decode_text(cifer, passphrase)
    print(cleartext)
