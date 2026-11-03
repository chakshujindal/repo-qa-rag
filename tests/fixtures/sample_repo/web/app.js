// Browser helpers for the bookshop storefront.

const currencyFormat = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

// Show a price like 1234.5 as "$1,234.50".
export function formatPrice(amount) {
  return currencyFormat.format(amount);
}

// Draw the cart summary into the page.
export function renderCart(cart, element) {
  const rows = Object.entries(cart.items).map(
    ([isbn, qty]) => `<li>${isbn} x ${qty}: ${formatPrice(cart.prices[isbn] * qty)}</li>`
  );
  element.innerHTML = `<ul>${rows.join("")}</ul>`;
}
