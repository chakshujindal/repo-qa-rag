"""Card payments through an external payment provider."""

import time


class PaymentError(Exception):
    pass


class PaymentGateway:
    """Thin client for the card processor's HTTP API."""

    def __init__(self, http, api_key: str, attempts: int = 3) -> None:
        self.http = http
        self.api_key = api_key
        self.attempts = attempts

    def charge(self, order_id: str, amount_cents: int, card_token: str) -> str:
        """Charge a card. Timeouts are retried with a short pause; returns the charge id."""
        for attempt in range(self.attempts):
            try:
                response = self.http.post(
                    "/charges",
                    json={"order": order_id, "amount": amount_cents, "source": card_token},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                return response["id"]
            except TimeoutError:
                time.sleep(2**attempt)
        raise PaymentError(f"charge for order {order_id} timed out")

    def refund(self, charge_id: str, amount_cents: int | None = None) -> str:
        """Refund a charge, fully or partially, returning money to the customer's card."""
        body = {"charge": charge_id}
        if amount_cents is not None:
            body["amount"] = amount_cents
        response = self.http.post("/refunds", json=body, timeout=10)
        return response["id"]
