# LLM_API

Call the API, and make use of the reasoning capability of LLM

## Running Tips

1. `pip install -e .`

2. 

~~~py
from llm_api import *

llm_api(
    messages=['hello', QUERY_PLACEHOLDER, 'hello?', QUERY_PLACEHOLDER],
    model='gpt-3.5-turbo', print_messages=True,
)
~~~

<!-- 3. `llm_api()
1. edit `src/template_script/001-pdtb3_top_subtext.py`

2. run `sh run.sh 001`

3. run `python src/process_pred.py` -->
