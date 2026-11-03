"""Shopping cart."""

from dataclasses import dataclass, field

COUPONS = {"WELCOME10": 0.10, "SUMMER25": 0.25}


@dataclass
class Cart:
    """Items a customer intends to buy, keyed by ISBN."""

    items: dict[str, int] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    discount: float = 0.0

    def add_item(self, isbn: str, price: float, quantity: int = 1) -> None:
        self.items[isbn] = self.items.get(isbn, 0) + quantity
        self.prices[isbn] = price

    def remove_item(self, isbn: str) -> None:
        self.items.pop(isbn, None)
        self.prices.pop(isbn, None)

    def subtotal(self) -> float:
        return sum(self.prices[isbn] * qty for isbn, qty in self.items.items())

    def total(self) -> float:
        return round(self.subtotal() * (1 - self.discount), 2)


def apply_coupon(cart: Cart, code: str) -> bool:
    """Apply a promotional coupon code to the cart; unknown codes are rejected."""
    rate = COUPONS.get(code.upper())
    if rate is None:
        return False
    cart.discount = max(cart.discount, rate)
    return True
