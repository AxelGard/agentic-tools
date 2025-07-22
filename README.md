# agentic-tools
A python package that lets you setup agents with tools very easy 

<img src="https://raw.githubusercontent.com/AxelGard/agentic-tools/master/docs/icon.png" alt="drawing" style="width:300px;"/>


All you need is to add a docorator do the function that should be made accessable to the LLM. 

```python
from agentic_tools import ai_tool

def my_function_that_an_llm_can_not_use():
    pass 

@ai_tool
def my_function_that_an_llm_can_use():
    pass

```

Be awere that this will eat more input tokens, since you are adding context of the functions.

## install 

```bash 
pip install agentic-tools 
```

Agentic tools needs a [LangChain](https://python.langchain.com/docs/integrations/chat/) object chat model.


### build 

```bash 
git clone git@github.com:AxelGard/agentic-tools.git
cd agentic-tools
python3 -m venv env 
source ./env/bin/activate
pip install -e .
# or 
# pip install -e .[dev] 
# if you want dev depenacies 
```


## DEMO 

you can check out [my expriment notebook](https://github.com/AxelGard/agentic-tools/blob/master/expr.ipynb)

or

suppose you have some functions that you want you LLM to be able to call. 
Agentic tools allows you to just add a decorator to it and then the `agent` will have access to those tools. 

All [LangChain](https://python.langchain.com/docs/integrations/chat/) models are supported, but for this demo I will use [Ollama](https://ollama.com/) with [llama3.1](https://ollama.com/library/llama3.1)

### install 

install Ollma 

```bash 
curl -fsSL https://ollama.com/install.sh | sh
```
and then run ollama  
```bash 
ollama run llama3.1
```

install the python dependacnices

```bash 
pip install agentic-tools langchain-ollama yfinance
```

### Example 

```python

from langchain_ollama.chat_models import ChatOllama
from agentic_tools import ai_tool, Agent
import yfinance as yf


@ai_tool
def get_stock_price(symbol: str, period:str="1d") -> str:
    stock = yf.Ticker(symbol)
    price = stock.history(period=period)["Close"].iloc[0]
    return f"The stock price of {symbol} was ${price:.2f} {period} ago"

@ai_tool
def get_market_cap(symbol: str) -> str:
    stock = yf.Ticker(symbol)
    cap = stock.info.get("marketCap", None)
    if cap:
        return f"The market cap of {symbol} is ${cap:,}"
    return f"Market cap data for {symbol} is not available."

@ai_tool
def get_pe_ratio(symbol: str) -> str:
    """ Will return P/E ratrio of the company """
    stock = yf.Ticker(symbol)
    pe = stock.info.get("trailingPE", None)
    if pe:
        return f"The P/E ratio of {symbol} is {pe:.2f}"
    return f"P/E ratio data for {symbol} is not available."


llm = ChatOllama(model="llama3.1", temperature=0)
agent = Agent(llm_chat_model=llm)

query = "what was apples stock price a 5 days ago?"
print(f"{agent.query(question=query)}\n")

query="who are you?"
print(f"{agent.query(question=query)}\n")

```