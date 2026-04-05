# The Market’s Sensory System 

Most people think financial data is just a list of numbers. It’s not. It’s a messy, noisy, and often broken reflection of human psychology. If you feed those broken numbers into a "smart" trading algorithm, you don't get a smart trader—you get an expensive catastrophe.

This is a pipeline designed to solve the **"Garbage In, Garbage Out"** problem in Quantitative Finance.

##  The Philosophy: "What I cannot create, I do not understand."

I built this because I wanted to see exactly where the math breaks. When a stock price hits zero, or a connection drops, or a "fat-tail" event occurs, most software just panics and crashes. This engine is built with **Feynman Shields**—mathematical guards that allow the system to keep its head when the data loses its mind.



##  How the Machinery Works

Instead of one giant, complicated script, I broke the problem into small pieces that each do one thing perfectly:

1. **The Bouncer (Validation):** Before a single calculation happens, we check the physical laws. If a "High" price is lower than a "Low" price, that’s not a market—that’s a glitch. We kick it out.
2. **The Gap Filler:** Markets don't wait for your internet connection. If a minute of data is missing, we don't just ignore it; we bridge it logically so the timeline stays intact.
3. **The Returns Engine:** This is the core. It calculates the heartbeat (returns) and the fear (volatility). It’s shielded against divide-by-zero errors and infinite loops.
4. **The Stress Test:** There are 148 automated tests in here. They act like a specialized laboratory, throwing fake crashes and corrupted data at the engine to prove it won't break when real money is on the line.

##  Getting the Engine Running

If you want to see the machinery in action, follow these steps:

### 1. Assemble the Parts
```bash
git clone [https://github.com/sumitmasirkar01/Financial-data-pipeline.git](https://github.com/sumitmasirkar01/Financial-data-pipeline.git)
cd Financial-data-pipeline
pip install -r requirements.txt
