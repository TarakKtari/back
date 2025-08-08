import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
from flask_socketio import SocketIO

# Initialize global variables
rates = {
    'USD': {'XE': 0, 'WISE': 0, 'YahooFinance': 0},
    'EUR': {'XE': 0, 'WISE': 0, 'YahooFinance': 0},
    'JPY': {'XE': 0, 'WISE': 0, 'YahooFinance': 0}
}
metric = {
    'USD': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0},
    'EUR': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0},
    'JPY': {"High": 0, "Low": 0, "Average": 0, "Volatility": 0}
}
rates_all = {
    'EUR/USD': {'rate': 0, 'change': ''},
    'USD/JPY': {'rate': 0, 'change': ''},
    'GBP/USD': {'rate': 0, 'change': ''},
    'EUR/JPY': {'rate': 0, 'change': ''},
    'GBP/EUR': {'rate': 0, 'change': ''},
    'USD/CHF': {'rate': 0, 'change': ''}
}
lastUpdated = None
socketio = None

def update_currency_rates(currency):
    global rates, lastUpdated
    urlTemplateXE = "https://www.xe.com/currencyconverter/convert/?Amount=1&From={}&To=TND"
    urlTemplateWISE = "https://wise.com/us/currency-converter/{}-to-tnd-rate?amount=1"
    urlXE = urlTemplateXE.format(currency)
    urlWISE = urlTemplateWISE.format(currency)

    try:
        response = requests.get(urlXE)
        soup = BeautifulSoup(response.content, "html.parser")
        xe_rate_element = soup.select_one(".result__BigRate-sc-1bsijpp-1.dPdXSB")
        if xe_rate_element:
            rates[currency]['XE'] = re.sub(r"[^\d\-.]", "", xe_rate_element.get_text())
        else:
            print(f"XE rate element not found for {currency}")

        high_value = soup.select_one('th:contains("High") + td')
        if high_value:
            metric[currency]["High"] = high_value.text.strip()

        low_value = soup.select_one('th:contains("Low") + td')
        if low_value:
            metric[currency]["Low"] = low_value.text.strip()

        volatility_value = soup.select_one('th:contains("Volatility") + td')
        if volatility_value:
            metric[currency]["Volatility"] = volatility_value.text.strip()
    except Exception as e:
        print(f"Error scraping XE.com for {currency}: {e}")

    try:
        response = requests.get(urlWISE)
        soup = BeautifulSoup(response.content, "html.parser")
        wise_rate_element = soup.select_one(".text-success")
        if wise_rate_element:
            rates[currency]['WISE'] = re.sub(r"[^\d\-.]", "", wise_rate_element.get_text())
    except Exception as e:
        print(f"Error scraping Wise.com for {currency}: {e}")

    try:
        rate_values = [float(rates[currency].get('XE', 0)), float(rates[currency].get('WISE', 0))]
        average_rate = sum(rate_values) / len(rate_values)
        metric[currency]['Average'] = average_rate
    except Exception as e:
        print(f"Error calculating average rate for {currency}: {e}")

    try:
        response = requests.get("https://www.xe.com/currencycharts/")
        soup = BeautifulSoup(response.content, "html.parser")
        currency_table = soup.find_all("table", class_="table__TableBase-sc-1j0jd5l-0")[0]
        rows = currency_table.find_all("tr")[1:]

        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 3:
                pair_link = cells[0].find('a')
                if pair_link:
                    pair_text = pair_link.get_text(strip=True)
                    rate_text = cells[1].get_text(strip=True)
                    change_symbol = cells[2].text.strip()
                    change_direction = 'Stable'
                    if '▲' in change_symbol:
                        change_direction = 'Up'
                    elif '▼' in change_symbol:
                        change_direction = 'Down'

                    for pair in rates_all.keys():
                        if pair.replace("/", " / ") == pair_text:
                            rates_all[pair]['rate'] = float(re.sub(r"[^\d.]", "", rate_text))
                            rates_all[pair]['change'] = change_direction
    except Exception as e:
        print(f"Error all currency: {e}")

    lastUpdated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if socketio:
        socketio.emit('rates_update', {
            'currency': currency,
            'rates': rates[currency],
            'lastUpdated': lastUpdated,
            'metrics': metric[currency],
            'rates_all': rates_all
        })
    else:
        print("SocketIO not initialized.")