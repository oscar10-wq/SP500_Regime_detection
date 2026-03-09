# Data Dictionary and Feature Definitions

This section outlines the financial, momentum, and macroeconomic variables used in our regime detection model.

---

# S&P 500 Market Data

## SPX_Close
The final trading price of the S&P 500 index at the end of a given period. It provides a baseline snapshot of overall U.S. large-cap stock market performance.

---

## SPX_Volume
Total trading activity in S&P 500 constituents measuring market participation and conviction.

- High volume → uncertainty, information arrival, or regime transition  
- Low volume → stable macro environment or complacency  
- Volume spikes often occur around policy or economic shocks  

---

## SPX_ROC (Rate of Change)

A momentum indicator that measures the percentage change in price between the current period and a past period.

- Positive momentum → improving growth expectations and expansion regime  
- Negative momentum → deteriorating outlook and economic slowdown risk  
- Reflects persistence of macroeconomic cycles  

$$
ROC = \frac{Close_{current} - Close_{previous}}{Close_{previous}} \times 100
$$

---

## SPX_RSI (Relative Strength Index)

Bounded oscillator (0–100) measuring buying vs selling pressure and market sentiment extremes, i.e., the speed and change of price movements on a scale of 0 to 100.

- High RSI → excessive optimism or potential asset overvaluation  
- Low RSI → panic, stress, or deleveraging conditions  
- Extremes often precede macro turning points  

Values above **70** typically indicate **overbought** conditions, while values below **30** indicate **oversold** conditions.

$$
RSI = 100 - \left( \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}} \right)
$$

---

## SPX_MACD (Moving Average Convergence Divergence)

SPX_MACD, SPX_MACDH, SPX_MACDS (Moving Average Convergence Divergence)

The MACD indicates the relationship between a longer-term moving average and a short-term moving average.

- **SPX_MACD** — Core line calculated from the difference between the 26-day EMA and the 12-day EMA  
- **SPX_MACDS** — Signal line (9-day EMA of MACD) that functions as a buy or sell trigger  
- **SPX_MACDH** — Histogram visualizing the difference between the MACD and the signal line  

$$
MACD = EMA_{12} - EMA_{26}
$$

---

# Volatility and Macroeconomic Data

## VIX_Close

The closing price of the **CBOE Volatility Index (VIX)**.

It measures the stock market's expectation of volatility based on S&P 500 index options. It is commonly referred to as the **"fear gauge"** because it spikes when investors anticipate market turbulence.

---

## Real_GDP

Real Gross Domestic Product is the total monetary value of all goods and services produced within a country, adjusted for inflation.

It provides a broad, intuitive measure of a country's **true economic growth and overall health**.

---

## Unemployment

Represents the percentage of the total labor force that is unemployed but actively seeking employment.

- High unemployment → economic distress  
- Low unemployment → strong economic conditions  

---

## Inflation

Represented by the **Core PCE (Personal Consumption Expenditures) price index**.

This measures the rate at which the general level of prices for goods and services is rising, excluding volatile food and energy sectors. It dictates how fast the purchasing power of money is eroding.

---

# Interest Rates and Futures

## Fed_Funds_Rate

The target interest rate set by the **Federal Reserve** at which commercial banks borrow and lend their excess reserves to each other overnight.

It is the **primary monetary policy tool** used by the Federal Reserve to control economic growth.

---

## 10Y2Y_Spread

Represents the difference in yield between the **10-year Treasury note** and the **2-year Treasury note**.

- Positive spread → normal economic conditions  
- Negative spread (inversion) → widely considered a **leading indicator of recession**

---

## Fed_Funds_Future

Financial contracts representing the market's expectation of what the daily official Federal Funds Rate will be at the time of contract expiry.

These contracts effectively represent **investor bets on future Federal Reserve monetary policy**.

---

## 10Y_Treasury_Future

Futures contracts tied to the **10-year U.S. Treasury note**.

Price fluctuations provide insight into:

- Long-term interest rate expectations  
- Broader economic outlook  

---

# Definitions

## What is a Treasury?

A Treasury note is essentially an **official IOU from the United States government**.

When the government needs to borrow money to fund infrastructure or public services, it sells these notes to investors. Investors lend money to the government and receive:

- Periodic interest payments
- The full principal repayment at maturity

Because they are backed by the U.S. government, Treasury securities are considered **one of the safest financial assets**.

---

### Example

10-year Treasury note for **$1,000** paying **4% interest**.

Interest is typically paid semi-annually.

$$
\text{Annual Payment} = \text{Principal} \times \text{Interest Rate}
$$

$$
\text{Annual Payment} = \$1000 \times 0.04 = \$40
$$

At the end of the **10-year maturity**, the investor receives the original **$1,000 principal** in addition to all interest payments already received.

---

## What is a Treasury Future?

A **10-year Treasury note** is a long-term government IOU that pays a fixed interest rate.

If the Federal Reserve raises interest rates:

- Newly issued bonds pay **higher interest**
- Older bonds with **lower yields become less attractive**

To sell an older bond, its price must fall.

This creates the fundamental inverse relationship:

$$
\text{Higher Interest Rates} \implies \text{Lower Bond Prices}
$$

When the project tracks **10-year Treasury Futures**, it is observing investors trading contracts based on the **future price of these government bonds**.

Predicting bond price movements effectively means predicting **future Federal Reserve policy decisions**.

Tracking these market expectations gives the regime detection model a **critical early signal** about whether the economy is expected to expand or contract.

---

# How Each Metric / Feature Interacts

The cycle begins with the **broad economy**, measured by:

- Real_GDP
- Unemployment

When the economy runs too hot and employment is booming, **inflation rises**.  

To control inflation, the **Federal Reserve raises the Fed_Funds_Rate**.

This action becomes the **primary trigger** in the regime detection framework because higher interest rates increase borrowing costs and reduce both corporate investment and consumer spending.

---

## Bond Market Reaction

Before the stock market reacts, the **bond market anticipates Federal Reserve policy**.

Investors use:

- Fed_Funds_Future  
- 10Y_Treasury_Future  

to place bets on how aggressively the Fed will raise rates.

As short-term rates rise faster than long-term growth expectations, the **10Y2Y_Spread inverts**.

---

## Market Stress Phase

Once money becomes expensive and the yield curve inverts:

- Market fear increases  
- **VIX_Close spikes**  
- Equity markets begin selling off  

This causes:

- Falling **SPX_Close**
- Rising **SPX_Volume**

This is the mechanical transition from a **Bull Market Regime** to a **Bear Market Regime**.

**Note:**  
The sell-off occurs because investors anticipate that higher borrowing costs and weaker consumer demand will reduce corporate earnings.

$$
\text{Rising Inflation}
\implies
\text{Fed Rate Hikes}
\implies
\text{Yield Curve Inversion}
\implies
\text{Bear Market Regime}
$$

---

## Technical Confirmation Signals

While macroeconomic indicators explain **why** the regime is changing, technical indicators identify **when the momentum shift occurs**.

As the S&P 500 declines:

- **SPX_ROC becomes negative**
- **SPX_RSI moves toward oversold territory**
- **SPX_MACD crosses below SPX_MACDS**
- **SPX_MACDH shrinks**

Macro variables provide the **early warning system**, while technical indicators provide the **precise timing signals** for regime transitions.