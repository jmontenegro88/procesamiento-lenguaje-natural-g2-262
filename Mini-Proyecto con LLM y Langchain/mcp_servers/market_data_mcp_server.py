from mcp.server.fastmcp import FastMCP
import httpx
import re

mcp = FastMCP("MercadoFinancieroMCP", host="localhost", port=9084)

# Mapeo de nombres comunes de activos a sus tickers de mercado
NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "sp500": "^GSPC",
    "s&p500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
}

# Mapeo de nombres de monedas a codigos ISO 4217
CURRENCY_ALIASES = {
    "dolar": "USD",
    "dólar": "USD",
    "dolares": "USD",
    "dólares": "USD",
    "usd": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "libra": "GBP",
    "libras": "GBP",
    "gbp": "GBP",
    "yen": "JPY",
    "yenes": "JPY",
    "jpy": "JPY",
    "franco": "CHF",
    "chf": "CHF",
    "dolar canadiense": "CAD",
    "cad": "CAD",
    "dolar australiano": "AUD",
    "aud": "AUD",
    "real": "BRL",
    "reales": "BRL",
    "brl": "BRL",
}


def _normalize_ticker(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    lower = symbol.strip().lower()
    if lower in NAME_TO_TICKER:
        return NAME_TO_TICKER[lower]
    # Remover signos como $, comillas o parentesis
    cleaned = re.sub(r"[\$\(\)\"\'\s]", "", cleaned)
    return cleaned if cleaned else "AAPL"


def _normalize_currency(curr: str) -> str:
    lower = curr.strip().lower()
    if lower in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[lower]
    cleaned = curr.strip().upper()
    cleaned = re.sub(r"[^A-Z]", "", cleaned)
    return cleaned if len(cleaned) == 3 else "USD"


@mcp.tool()
def consultar_cotizacion(symbol: str) -> str:
    """Consulta la cotizacion de mercado mas reciente para una accion, indice o criptomoneda (e.g. AAPL, MSFT, NVDA, TSLA, BTC-USD)."""
    ticker = _normalize_ticker(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5d&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = httpx.get(url, headers=headers, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return f"No se encontro informacion financiera para el activo: '{symbol}' (ticker: {ticker})."

        meta = result[0].get("meta", {})
        regular_price = meta.get("regularMarketPrice")
        currency = meta.get("currency", "USD")
        prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")
        high_day = meta.get("regularMarketDayHigh")
        low_day = meta.get("regularMarketDayLow")

        if regular_price is None:
            indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
            closes = [c for c in indicators.get("close", []) if c is not None]
            if closes:
                regular_price = round(closes[-1], 2)
            else:
                return f"Cotizacion no disponible actualmente para {ticker}."

        diff_str = ""
        if prev_close and prev_close > 0:
            change = regular_price - prev_close
            change_pct = (change / prev_close) * 100
            sign = "+" if change >= 0 else ""
            diff_str = f", variacion diaria: {sign}{change:.2f} {currency} ({sign}{change_pct:.2f}%)"

        rango_str = ""
        if high_day is not None and low_day is not None:
            rango_str = f", rango diario: [{low_day:.2f} - {high_day:.2f}]"

        return (
            f"Cotizacion de {ticker}: precio actual {regular_price:.2f} {currency}"
            f"{diff_str}{rango_str}. (Fuente: Yahoo Finance)."
        )
    except httpx.HTTPStatusError as exc:
        return f"Error al consultar cotizacion de {ticker}: estado HTTP {exc.response.status_code}."
    except Exception as exc:
        return f"Error de conexion al consultar {ticker}: {exc}."


@mcp.tool()
def tasa_cambio(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Consulta la tasa de cambio oficial entre dos divisas (e.g. USD, EUR, GBP, JPY, CAD, BRL) y calcula la conversion para un monto dado."""
    from_clean = _normalize_currency(from_currency)
    to_clean = _normalize_currency(to_currency)

    if from_clean == to_clean:
        return f"Tasa de cambio 1:1 entre {from_clean} y {to_clean}. Monto convertido: {amount:.2f} {to_clean}."

    url = f"https://api.frankfurter.dev/v1/latest?base={from_clean}&symbols={to_clean}"
    try:
        response = httpx.get(url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        rates = data.get("rates", {})
        if to_clean not in rates:
            return f"No se encontro tasa de cambio entre {from_clean} y {to_clean}."

        rate = rates[to_clean]
        converted = amount * rate
        date_str = data.get("date", "reciente")

        return (
            f"Tasa de cambio oficial ({date_str}): 1 {from_clean} = {rate:.4f} {to_clean}. "
            f"Para {amount:,.2f} {from_clean}, el monto equivalente es {converted:,.2f} {to_clean}. "
            f"(Fuente: Banco Central Europeo / Frankfurter)."
        )
    except httpx.HTTPStatusError as exc:
        return f"Error en API de divisas ({from_clean} -> {to_clean}): HTTP {exc.response.status_code}."
    except Exception as exc:
        return f"Error de conexion al consultar divisas {from_clean} a {to_clean}: {exc}."


if __name__ == "__main__":
    # Servidor HTTP en el puerto 9084 para compatibilidad con Colab y local
    mcp.run(transport="streamable-http")
