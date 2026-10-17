from mcp.server.fastmcp import FastMCP
import httpx
import math
import re

mcp = FastMCP("AnaliticaFinancieraMCP", host="localhost", port=9085)

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
}


def _clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    lower = symbol.strip().lower()
    if lower in NAME_TO_TICKER:
        return NAME_TO_TICKER[lower]
    cleaned = re.sub(r"[\$\(\)\"\'\s]", "", cleaned)
    return cleaned if cleaned else "AAPL"


@mcp.tool()
def calcular_rendimiento_y_riesgo(symbol: str, dias: int = 30) -> str:
    """Calcula metricas cuantitativas de rendimiento acumulado, volatilidad diaria/anualizada y ratio Sharpe para un activo en una ventana de dias (e.g. 15, 30, 60, 90)."""
    ticker = _clean_symbol(symbol)
    dias = max(5, min(int(dias), 365))

    # Mapeo de dias a rango de Yahoo Finance
    if dias <= 7:
        yf_range = "7d"
    elif dias <= 30:
        yf_range = "1mo"
    elif dias <= 90:
        yf_range = "3mo"
    elif dias <= 180:
        yf_range = "6mo"
    else:
        yf_range = "1y"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={yf_range}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = httpx.get(url, headers=headers, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return f"No se obtuvieron datos historicos para el activo {ticker}."

        quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
        raw_closes = quotes.get("close", [])
        closes = [float(p) for p in raw_closes if p is not None and p > 0]

        if len(closes) < 3:
            return f"Datos insuficientes ({len(closes)} observaciones) para calcular metricas de {ticker}."

        # Tomar los ultimos `dias` observaciones si hay mas
        closes = closes[-dias:]
        p_inicial = closes[0]
        p_final = closes[-1]
        p_max = max(closes)
        p_min = min(closes)

        # Retornos diarios
        daily_returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        n = len(daily_returns)
        mean_ret = sum(daily_returns) / n
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (n - 1) if n > 1 else 0.0
        daily_vol = math.sqrt(variance)

        # Anualizacion (252 dias habiles bursatiles)
        ann_vol = daily_vol * math.sqrt(252)
        total_return = ((p_final - p_inicial) / p_inicial) * 100.0
        ann_return = mean_ret * 252 * 100.0

        # Sharpe ratio aproximado (asumiendo tasa libre de riesgo = 4.0% anual)
        rf = 4.0
        sharpe = (ann_return - rf) / (ann_vol * 100.0) if (ann_vol > 0) else 0.0

        return (
            f"Analisis Cuantitativo de {ticker} (ventana: {len(closes)} sesiones / ~{dias} dias):\n"
            f"- Precio inicial: {p_inicial:.2f} USD | Precio final: {p_final:.2f} USD\n"
            f"- Rango del periodo: minimo {p_min:.2f} USD, maximo {p_max:.2f} USD\n"
            f"- Rendimiento acumulado: {total_return:+.2f}%\n"
            f"- Retorno medio diario: {mean_ret * 100:+.3f}%\n"
            f"- Volatilidad diaria: {daily_vol * 100:.2f}%\n"
            f"- Volatilidad anualizada: {ann_vol * 100:.2f}%\n"
            f"- Ratio de Sharpe anualizado (Rf=4%): {sharpe:.2f}\n"
            f"Conclusion: {'Activo con tendencia alcista' if total_return > 0 else 'Activo con tendencia bajista'} "
            f"y perfil de riesgo {'alto' if ann_vol * 100 > 30 else 'moderado/bajo'}."
        )
    except httpx.HTTPStatusError as exc:
        return f"Error HTTP al consultar historico de {ticker}: {exc.response.status_code}."
    except Exception as exc:
        return f"Error en calculo cuantitativo de {ticker}: {exc}."


@mcp.tool()
def proyeccion_interes_compuesto(
    capital_inicial: float,
    tasa_anual_pct: float,
    anios: int,
    aporte_mensual: float = 0.0,
) -> str:
    """Calcula la proyeccion matematica exacta de una inversion a interes compuesto con capital inicial, tasa anual %, plazo en anios y aportes mensuales."""
    capital_inicial = float(capital_inicial)
    tasa_anual_pct = float(tasa_anual_pct)
    anios = max(1, int(anios))
    aporte_mensual = float(aporte_mensual)

    r_mensual = (tasa_anual_pct / 100.0) / 12.0
    total_meses = anios * 12

    # Capitalizacion compuesta mes a mes
    saldo = capital_inicial
    hitos = {}
    for mes in range(1, total_meses + 1):
        saldo = saldo * (1.0 + r_mensual) + aporte_mensual
        if mes % 12 == 0:
            hitos[mes // 12] = saldo

    total_aportado = capital_inicial + (aporte_mensual * total_meses)
    total_intereses = saldo - total_aportado
    rentabilidad_total_pct = (total_intereses / total_aportado) * 100.0 if total_aportado > 0 else 0.0

    desglose_lineas = []
    # Mostrar hitos clave: ano 1, mitad y ano final
    anos_mostrar = sorted(list(set([1, max(1, anios // 2), anios])))
    for a in anos_mostrar:
        if a in hitos:
            desglose_lineas.append(f"  * Año {a}: {hitos[a]:,.2f} USD")

    desglose_str = "\n".join(desglose_lineas)

    return (
        f"Proyeccion de Interes Compuesto ({anios} años a {tasa_anual_pct:.2f}% anual):\n"
        f"- Capital inicial: {capital_inicial:,.2f} USD\n"
        f"- Aporte mensual: {aporte_mensual:,.2f} USD ({aporte_mensual * total_meses:,.2f} USD en {anios} años)\n"
        f"- Total propio invertido: {total_aportado:,.2f} USD\n"
        f"- Total ganado por interes compuesto: {total_intereses:,.2f} USD\n"
        f"- Saldo final acumulado: {saldo:,.2f} USD\n"
        f"- Ganancia porcentual sobre lo invertido: +{rentabilidad_total_pct:.2f}%\n"
        f"Evolucion proyectada:\n{desglose_str}"
    )


if __name__ == "__main__":
    # Servidor HTTP en el puerto 9085 para compatibilidad con Colab y local
    mcp.run(transport="streamable-http")
