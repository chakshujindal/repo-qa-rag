"""Order lifecycle: placing and cancelling orders."""

from bookshop.inventory import release_stock, reserve_stock
from bookshop.notifications import send_order_confirmation


class OrderError(Exception):
    pass


def place_order(cart, customer, stock, gateway, mailer, card_token: str) -> dict:
    """Checkout: reserve stock for every item, charge the card, then send a confirmation."""
    for isbn, quantity in cart.items.items():
        reserve_stock(stock, isbn, quantity)
    order = {"id": f"{customer['id']}-{len(cart.items)}", "items": dict(cart.items)}
    order["total"] = cart.total()
    order["charge_id"] = gateway.charge(order["id"], int(order["total"] * 100), card_token)
    order["status"] = "paid"
    send_order_confirmation(mailer, customer, order)
    return order


def cancel_order(order: dict, stock, gateway) -> dict:
    """Cancel an order that has not shipped yet: refund the charge and restock the items."""
    if order["status"] == "shipped":
        raise OrderError("shipped orders must be returned, not cancelled")
    gateway.refund(order["charge_id"])
    for isbn, quantity in order["items"].items():
        release_stock(stock, isbn, quantity)
    order["status"] = "cancelled"
    return order
