import bcrypt

# Function to hash a password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

# Function to verify a password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# Example usage
password = 'P@ssw0rd'
hashed_password = hash_password(password)
print(hashed_password)
# $2b$12$cu/MbGxRso7a.JZHZfSkQO9tTnDm1s3s3Eki8WYpA.LvDOd1bIuqO

is_valid_password = verify_password(password, hashed_password)
print(is_valid_password)
# True
