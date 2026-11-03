"""Customer emails."""

ORDER_TEMPLATE = """Hi {name},

Thanks for your order #{order_id}. We will ship {count} book(s) to:
{address}

Total charged: {total}
"""


def render_email(template: str, **fields: object) -> str:
    return template.format(**fields)


def send_order_confirmation(mailer, customer: dict, order: dict) -> None:
    """Email the customer a receipt with the order number, shipping address and total."""
    body = render_email(
        ORDER_TEMPLATE,
        name=customer["name"],
        order_id=order["id"],
        count=len(order["items"]),
        address=customer["address"],
        total=order["total"],
    )
    mailer.send(to=customer["email"], subject=f"Order #{order['id']} confirmed", body=body)
