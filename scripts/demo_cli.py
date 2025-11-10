# Demo CLI script

import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.typo_corrector import TypoCorrector

def main():
    corrector = TypoCorrector()
    
    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == 'exit':
            break
        result = corrector.correct(query)
        print(f"Original:  {result['original_query']}")
        print(f"Corrected: {result['normalized_query']}")
        print(f"Changed:   {result['changed']}")
        if result['tokens']:
            print("Tokens:")
            for token in result['tokens']:
                print(f"  - {token['original']} -> {token['corrected']} ({token['confidence']})")
        print()

if __name__ == "__main__":
    main()