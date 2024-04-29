from textwrap import dedent
import tiktoken
from numpy import mean
from plotly import graph_objects as go
from models import Evaluation

def get_num_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def message(role, content):
    return {
        "role": role,
        "content": dedent(content).strip()
    }

def system(content):
    return message("system", content)

def user(content):
    return message("user", content)

def assistant(content):
    return message("assistant", content)

def plot_success_rate(weak_evals, med_evals, strong_evals):
    weak_success_rate = mean([e.success for e in weak_evals])
    med_success_rate = mean([e.success for e in med_evals])
    strong_success_rate = mean([e.success for e in strong_evals])

    # Create a bar chart
    fig = go.Figure(data=[
        go.Bar(name='GPT-3.5-turbo', x=['Weak'], y=[weak_success_rate]),
        go.Bar(name='GPT-4-0613', x=['Medium'], y=[med_success_rate]),
        go.Bar(name='GPT-4-turbo', x=['Strong'], y=[strong_success_rate])
    ])
    
    # Update layout for clarity
    fig.update_layout(
        title='Success Rate by Evaluator Strength',
        xaxis_title='Evaluator Strength',
        yaxis_title='Success Rate',
        yaxis=dict(tickformat=".0%"),
    )

    fig.show() 