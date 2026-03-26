"""
RAG-based brand name lookup module for the correction pipeline.

Provides O(1) exact lookup and fuzzy matching for brand names, abbreviations,
units, and currencies. Prevents false corrections of legitimate e-commerce terms.
"""

from typing import Optional, Tuple, List, Set
import re


class BrandLookup:
    """
    RAG-based brand and entity lookup for correction pipeline.

    Maintains hash sets for O(1) lookups of brands, abbreviations, units, and
    currencies. Supports fuzzy matching using Levenshtein distance for near-misses.
    """

    def __init__(self):
        """Initialize brand lookup with comprehensive entity databases."""
        # Core brand names - electronics, fashion, grocery
        self.brands = {
            # Electronics
            "ASUS", "OPPO", "POCO", "REALME", "VIVO", "SAMSUNG", "LG", "SONY",
            "INTEL", "AMD", "NVIDIA", "BOSCH", "DYSON", "NIKON", "CANON", "GoPRO",
            "RAZER", "CORSAIR", "STEELSERIES", "LOGITECH", "DELL", "HP", "LENOVO",
            "APPLE", "MICROSOFT", "GOOGLE", "META", "QUALCOMM", "MEDIATEK",
            # Fashion
            "SHEIN", "H&M", "ZARA", "GUCCI", "PRADA", "LOUIS VUITTON", "NIKE",
            "ADIDAS", "PUMA", "REEBOK", "NEW BALANCE", "FILA", "LACOSTE",
            "RALPH LAUREN", "TOMMY HILFIGER", "CALVIN KLEIN", "UNIQLO",
            # Home & Kitchen
            "IKEA", "ZWILLING", "TRAMONTINA", "KITCHENAID", "DYSON", "DYSON",
            "VITAMIX", "INSTANT POT", "LE CREUSET", "LODGE", "OXLEY",
            # Grocery & Food
            "NESTLÉ", "COCA COLA", "PEPSI", "KRAFT", "HEINZ", "KELLOGG'S",
            "GENERAL MILLS", "MONDELEZ", "DANONE", "YOPLAIT", "HÄAGEN-DAZS",
            "LINDT", "GODIVA", "FERRERO", "MARS", "SNICKERS", "M&M'S",
            # Automotive
            "BOSCH", "MICHELIN", "GOODYEAR", "BRIDGESTONE", "DUNLOP", "PIRELLI",
            "SHELL", "MOBIL", "CASTROL", "VALVOLINE", "MOTUL",
            # Toys & Games
            "LEGO", "MATTEL", "HASBRO", "FISHER PRICE", "BARBIE", "HOT WHEELS",
            # Sports
            "DECATHLON", "SPALDING", "WILSON", "PING", "TITLEIST",
        }

        # Common abbreviations in e-commerce
        self.abbreviations = {
            # Hardware
            "GPU", "CPU", "RAM", "SSD", "HDD", "GPU", "PSU", "MOB", "GPU",
            "RTX", "GTX", "RX", "GFX", "VRAM", "LPDDR", "DDR3", "DDR4", "DDR5",
            # Connectivity
            "USB", "HDMI", "VGA", "DVI", "DP", "MINI DP", "TYPE C", "USBC",
            "WIFI", "BT", "4G", "5G", "LTE", "3G", "NFC", "RFID",
            # Storage
            "GB", "TB", "MB", "KB", "GBS", "TBS", "MBS", "KBS",
            # Other tech
            "FPS", "HZ", "KHZ", "MHZ", "GHZ", "DPI", "RPM", "TPM", "SIM",
            "LED", "OLED", "LCD", "QHD", "FHD", "UHD", "4K", "8K",
            # Commerce
            "PCS", "PIECES", "QTY", "DOZ", "PKG", "PKT", "BOX", "SET",
            "PACK", "CARTON", "CASE", "LOT", "BULK",
            # Measurements
            "PX", "PT", "IN", "FT", "YD", "MI", "KM", "M", "CM", "MM",
            # Units of weight
            "KG", "G", "LBS", "OZ", "TON", "MG",
            # Volume
            "L", "ML", "GAL", "QUART", "PINT", "CUP", "FL OZ",
            # Medical/Health
            "MG", "ML", "ML", "IU", "MCG", "MCGS",
        }

        # Units of measurement
        self.units = {
            # Metric
            "KG", "G", "ML", "L", "M", "CM", "MM", "KM",
            # Imperial
            "LBS", "OZ", "TON", "IN", "FT", "YD", "MI",
            # Volume
            "ML", "L", "GAL", "QUART", "PINT", "CUP", "TBSP", "TSP", "FL OZ",
            # Other
            "GHZ", "MHZ", "HZ", "DPI", "PPI", "RPM", "PSI", "BAR", "WATTS", "AMPS",
        }

        # Currency symbols and codes
        self.currencies = {
            # Symbols
            "$", "€", "£", "¥", "₹", "₺", "₽", "₩", "₪", "₦", "₨", "₱",
            "₡", "₲", "₵", "₴", "₸", "₹", "₺", "₽", "฿", "៛", "₾", "₿",
            # Codes
            "USD", "EUR", "GBP", "JPY", "INR", "CNY", "AUD", "CAD", "CHF",
            "SEK", "NZD", "MXN", "SGD", "HKD", "NOK", "KRW", "TRY", "RUB",
            "BRL", "ZAR", "AED", "SAR", "QAR", "KWD", "BHD", "OMR",
        }

        # Convert all to uppercase for case-insensitive matching
        self.brands = {brand.upper() for brand in self.brands}
        self.abbreviations = {abbr.upper() for abbr in self.abbreviations}
        self.units = {unit.upper() for unit in self.units}
        self.currencies = {curr.upper() for curr in self.currencies}

        # Combined set for quick lookup
        self.all_protected = self.brands | self.abbreviations | self.units | self.currencies

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Integer distance (0 = identical)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def lookup(self, token: str) -> Optional[str]:
        """Look up a token in the protected entity database.

        Performs exact lookup first (case-insensitive), then fuzzy matching.

        Args:
            token: Token to look up (e.g., "ASUS", "gpu", "5kg")

        Returns:
            Canonical form of the token if found, None otherwise
        """
        if not token:
            return None

        # Normalize: extract potential entity from token
        normalized = token.upper().strip()

        # Exact match lookup
        if normalized in self.all_protected:
            return normalized

        # Check each category for exact matches
        if normalized in self.brands:
            return normalized
        if normalized in self.abbreviations:
            return normalized
        if normalized in self.units:
            return normalized
        if normalized in self.currencies:
            return normalized

        # Fuzzy matching with threshold
        max_distance = 2
        best_match = None
        best_distance = max_distance

        # Search in all protected entities
        for protected in self.all_protected:
            distance = self._levenshtein_distance(normalized, protected)
            if distance < best_distance:
                best_distance = distance
                best_match = protected

        if best_match and best_distance <= max_distance:
            return best_match

        return None

    def protect_brands(self, query: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """Identify and mask brand tokens in query.

        Returns masked query and list of (position, token, category) tuples.

        Args:
            query: Input query string

        Returns:
            Tuple of (masked_query, protected_tokens)
            where protected_tokens is list of (start_pos, original, category)
        """
        tokens = query.split()
        protected_tokens: List[Tuple[int, str, str]] = []
        masked_tokens = []
        char_position = 0

        for token in tokens:
            # Check for brand/currency/etc at start of token
            protected = self._extract_protected_prefix(token)

            if protected:
                entity, entity_type, remaining = protected
                protected_tokens.append((char_position, entity, entity_type))

                # Replace entity with placeholder
                mask = f"[{entity_type.upper()}_{len(protected_tokens)-1}]"
                if remaining:
                    masked_tokens.append(mask + remaining)
                else:
                    masked_tokens.append(mask)
            else:
                masked_tokens.append(token)

            char_position += len(token) + 1  # +1 for space

        masked_query = " ".join(masked_tokens)
        return masked_query, protected_tokens

    def _extract_protected_prefix(self, token: str) -> Optional[Tuple[str, str, str]]:
        """Extract protected prefix from token.

        Returns:
            Tuple of (entity, category, remaining_token) or None
        """
        # Try to match prefixes of increasing length
        for i in range(len(token), 0, -1):
            prefix = token[:i]
            match = self.lookup(prefix)

            if match:
                # Determine category
                if match in self.brands:
                    category = "brand"
                elif match in self.abbreviations:
                    category = "abbreviation"
                elif match in self.units:
                    category = "unit"
                elif match in self.currencies:
                    category = "currency"
                else:
                    category = "unknown"

                remaining = token[i:]
                return (match, category, remaining)

        return None

    def apply_brand_corrections(self, original_query: str, model_output: str) -> str:
        """Merge RAG corrections with model output.

        This preserves protected entities from the original query while applying
        model corrections to other parts.

        Args:
            original_query: Original user query
            model_output: Output from typo correction model

        Returns:
            Corrected query with protected entities preserved
        """
        # Tokenize both queries
        original_tokens = original_query.split()
        model_tokens = model_output.split()

        # If token counts differ significantly, return model output
        if abs(len(original_tokens) - len(model_tokens)) > 2:
            return model_output

        # Rebuild query, preserving protected entities
        result_tokens = []

        for i, orig_token in enumerate(original_tokens):
            if i >= len(model_tokens):
                result_tokens.append(orig_token)
                continue

            model_token = model_tokens[i]

            # Check if original token contains protected entity
            protected = self._extract_protected_prefix(orig_token)

            if protected:
                entity, category, remaining = protected
                # Preserve the protected entity, apply model correction to remaining
                if remaining:
                    # Try to find matching remaining in model output
                    if remaining.lower() in model_token.lower():
                        result_tokens.append(orig_token)  # Preserve entirely
                    else:
                        # Use model correction for the remaining part
                        model_remaining = model_token[len(entity):]
                        result_tokens.append(entity + model_remaining)
                else:
                    result_tokens.append(entity)
            else:
                # No protected entity, use model output
                result_tokens.append(model_token)

        return " ".join(result_tokens)

    def is_protected(self, token: str) -> bool:
        """Check if token is a protected entity.

        Args:
            token: Token to check

        Returns:
            True if token is protected, False otherwise
        """
        normalized = token.upper().strip()
        return normalized in self.all_protected or self.lookup(token) is not None

    def get_category(self, token: str) -> Optional[str]:
        """Get the category of a protected token.

        Args:
            token: Token to categorize

        Returns:
            Category name (brand, abbreviation, unit, currency) or None
        """
        normalized = token.upper().strip()

        if normalized in self.brands:
            return "brand"
        elif normalized in self.abbreviations:
            return "abbreviation"
        elif normalized in self.units:
            return "unit"
        elif normalized in self.currencies:
            return "currency"

        # Check fuzzy matches
        match = self.lookup(token)
        if match:
            if match in self.brands:
                return "brand"
            elif match in self.abbreviations:
                return "abbreviation"
            elif match in self.units:
                return "unit"
            elif match in self.currencies:
                return "currency"

        return None

    def protect_numeric_units(self, query: str) -> Tuple[str, List[str]]:
        """Protect numeric values with units (e.g., "5 kg", "32 gb").

        Args:
            query: Input query

        Returns:
            Tuple of (protected_query, list_of_protected_values)
        """
        # Pattern: number + optional space + unit
        pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)'

        protected_values = []
        masked_query = query

        for match in re.finditer(pattern, query):
            number = match.group(1)
            unit = match.group(2).upper()

            # Check if it's a known unit
            if unit in self.units or unit in self.abbreviations:
                full_value = match.group(0)
                placeholder = f"[NUM_UNIT_{len(protected_values)}]"
                protected_values.append(full_value)
                masked_query = masked_query.replace(full_value, placeholder, 1)

        return masked_query, protected_values

    def protect_prices(self, query: str) -> Tuple[str, List[str]]:
        """Protect currency amounts (e.g., "$99.99", "€50").

        Args:
            query: Input query

        Returns:
            Tuple of (protected_query, list_of_protected_values)
        """
        # Pattern: currency symbol/code + number
        currency_pattern = r'([\$€£¥₹₺\w]{1,3})\s*(\d+(?:\.\d+)?)'

        protected_values = []
        masked_query = query

        for match in re.finditer(currency_pattern, query):
            currency = match.group(1).upper()
            amount = match.group(2)

            # Check if it's a known currency
            if currency in self.currencies:
                full_value = match.group(0)
                placeholder = f"[PRICE_{len(protected_values)}]"
                protected_values.append(full_value)
                masked_query = masked_query.replace(full_value, placeholder, 1)

        return masked_query, protected_values

    def get_statistics(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with counts for each entity type
        """
        return {
            "total_brands": len(self.brands),
            "total_abbreviations": len(self.abbreviations),
            "total_units": len(self.units),
            "total_currencies": len(self.currencies),
            "total_protected": len(self.all_protected),
        }


# Module-level convenience functions
_default_lookup = None


def get_default_lookup() -> BrandLookup:
    """Get or create the default BrandLookup instance."""
    global _default_lookup
    if _default_lookup is None:
        _default_lookup = BrandLookup()
    return _default_lookup


def lookup(token: str) -> Optional[str]:
    """Convenience function for looking up a token."""
    return get_default_lookup().lookup(token)


def is_protected(token: str) -> bool:
    """Convenience function to check if token is protected."""
    return get_default_lookup().is_protected(token)


def protect_brands(query: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Convenience function to protect brands in query."""
    return get_default_lookup().protect_brands(query)


def apply_corrections(original_query: str, model_output: str) -> str:
    """Convenience function to merge corrections."""
    return get_default_lookup().apply_brand_corrections(original_query, model_output)


if __name__ == "__main__":
    # Example usage
    lookup_engine = BrandLookup()

    # Test exact lookup
    print("=== EXACT LOOKUPS ===")
    test_tokens = ["ASUS", "GPU", "kg", "usd", "OPPO", "nonexistent"]
    for token in test_tokens:
        result = lookup_engine.lookup(token)
        category = lookup_engine.get_category(token)
        print(f"{token:15} -> {str(result):15} (category: {category})")

    # Test fuzzy matching
    print("\n=== FUZZY MATCHING ===")
    fuzzy_tokens = ["ASSU", "KGG", "US", "USd"]
    for token in fuzzy_tokens:
        result = lookup_engine.lookup(token)
        print(f"{token:15} -> {result}")

    # Test brand protection
    print("\n=== BRAND PROTECTION ===")
    queries = [
        "ASUS laptop gaming",
        "5 kg protein powder",
        "$99.99 headphones",
        "qty 10 usb cables"
    ]
    for query in queries:
        masked, tokens = lookup_engine.protect_brands(query)
        print(f"Original: {query}")
        print(f"Masked:   {masked}")
        print(f"Protected: {tokens}\n")

    # Test numeric units
    print("=== NUMERIC UNITS ===")
    numeric_queries = ["32 gb ram", "5kg powder", "1000ml juice"]
    for query in numeric_queries:
        protected, values = lookup_engine.protect_numeric_units(query)
        print(f"{query:20} -> {protected:20} (protected: {values})")

    # Print statistics
    print("\n=== STATISTICS ===")
    stats = lookup_engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
