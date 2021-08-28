import os

client_id = os.getenv('TDAMERITRADE_CLIENT_ID')
account_id = os.getenv('TDAMERITRADE_ACCOUNT_ID')
refresh_token = os.getenv('R8Fv1ZMqycDsF7sau78v4KO4CK5jXLlmVL64+IJN2nUmi882wuIN8ULhY3vu0r/HPHUKvuaXRKTMyenKLMDXX9DYQFHjR6zISBtq+e33u8AyT3hOl0Zb47em2MFARZwEi7DhTCaZgxNRkGyiHIUOZmZ7oEfmHlKSPwhaRKFO+eSGNfvs+nSLZVU2UghCeFQcd4Ob4SGT6DXyICi9pH5xM2Nn18eL1XhJqfTERU+O/+1BRcjf1YXyukue1KybV+hNf14gaJiipmBRbcgNztN/xb5jxR6PJsTGQRZ4iXGCogA5CBkqQMlRdBqnHHLYwoMA9pdmO3IwYRh20dgUyfIpNvdZxwNWLCT2FmH8pOxSLr8RXhIGGz6HMa6vVIPp8cYSqfWnNAUDq1tnLFZtgn6nwsqOfiMbqJptn7D3YdaX1xf3kwkvQo8TqyKG09g100MQuG4LYrgoVi/JHHvlKOKf8BOn8jAHKc+Zwy3KZ11btvL+akMRIfIKYBNRIuepfGQbr3O0PBe56kbnyOS4d9jPKJgyPCj9kcwR8QO6Jq3yw5QNM2tceiYkWJEHU1EuEaQoUeVgcde9YXfP6j9uIWGVHYmc3ZMCa7HUxONBX6lY9PUYdW2Fuc+TzOtlnHX1y/nhUoYYYewfBnpbquZX891LrAqstjaTnobfK66pDpPRs6QCyQDJ36h5BFa7pcI7vilpC3l4aYJvfz1v+/VxvahGnutWR6Ne0Aipgu9OMt3PpObShwA333sngCfvO56tnMWmeeboqk3IE/enwCpKPMtADjLV1lYP3JIn+cd1PppSAMJzmm+CCh39v3Voo3I+sWZz0k9v86O6kDoCKmz3rvDSFnV1Mf8iyKJ24CuskR976D1hDxG22YUaAAn+/p57tXS1//ljWFxOV/s=212FD3x19z9sWBHDJACbC00B75E')

from td.client import TDClient

# Create a new session, credentials path is required.
TDSession = TDClient(
    client_id=os.getenv('TDAMERITRADE_CLIENT_ID'),
    redirect_uri=os.getenv('TDPY_CALLBACK_URL'),
    credentials_path='./tdameritrade-state.json'
)

# Login to the session
TDSession.login()

# Grab real-time quotes for 'MSFT' (Microsoft)
msft_quotes = TDSession.get_quotes(instruments=['MSFT'])

# Grab real-time quotes for 'AMZN' (Amazon) and 'SQ' (Square)
multiple_quotes = TDSession.get_quotes(instruments=['AMZN','SQ'])

