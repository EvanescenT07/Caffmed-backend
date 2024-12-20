from waitress import serve
from model import app
import os


if __name__ == '__main__':
      port = int(os.getenv('PORT', 5000))
      print(f"starting server on port {port}")
      serve(
            app,
            host='0.0.0.0',
            port=port,
            threads=4,
            url_scheme='http'
      )