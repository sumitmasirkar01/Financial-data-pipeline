# The Market’s Sensory System 

Most people think financial data is just a clean list of numbers. It isn’t. It’s a noisy, broken reflection of human psychology. If you feed those broken numbers into a "smart" trading algorithm, you don't get a smart trader—you get an expensive catastrophe.

I built `market-pipeline` to solve the **"Garbage In, Garbage Out"** problem. This isn't just a data scraper; it’s a **mathematical filter** that stands between the chaos of raw vendor APIs and the precision of your trading models.



##  The Philosophy: "What I cannot create, I do not understand."

When a stock price hits zero, or a connection drops, or a "fat-tail" event occurs, most software just panics and crashes—or worse, it stays silent and lets the error poison your entire database.

This pipeline is built with **Feynman Shields**—mathematical guards that allow the system to keep its head when the data loses its mind. It doesn't just "clean" data; it **audits** it. It gives every symbol a "Trust Grade," so you know exactly how much of your result is real signal and how much is just a bridge built over a gap.

##  How the Machinery Works

* **The Bouncer (Validator):** We check the physical laws first. If the "High" is lower than the "Low," that’s not a market; that’s a glitch. We flag it and move on.
* **The Timekeeper (Gaps & Calendars):** Markets don't follow a simple Monday-Friday clock; they follow exchange calendars (NYSE, NSE). We align the data to the heartbeat of the actual exchange, filling the silences logically so your timeline never skips a beat.
* **The Auditor (Adjustments):** Corporate actions like splits and dividends are never "automatic" here. They are explicit, typed inputs. We backward-adjust prices and volumes, leaving a clear audit trail of every cent accounted for.
* **The Risk Engine (Returns & VaR):** We calculate the heartbeat (returns) and the fear (Value at Risk). These are shielded against divide-by-zero errors and infinite loops.
* **The Crisis Detector (Outliers):** We don't use standard deviations (which get blinded by a single crash). We use **Rolling MAD** (Median Absolute Deviation) to find the true weirdos in the data without letting one bad day hide the rest.

##  The Laboratory (Quality & Rigor)

There are **148 automated tests** in this repo. They act like a stress-test laboratory, throwing fake crashes, corrupted OHLCV bars, and "Black Swan" events at the engine to prove it won't break when real money is on the line.

##  Getting Started

### 1. Assemble the Parts
```bash
git clone [https://github.com/sumitmasirkar01/Financial-data-pipeline.git](https://github.com/sumitmasirkar01/Financial-data-pipeline.git)
cd Financial-data-pipeline
pip install -r requirements.txt
