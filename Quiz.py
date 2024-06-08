import streamlit as st
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
import together
from typing import Any, Dict
from pydantic import Extra, root_validator
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
import json
from langchain.chat_models import ChatOpenAI
from typing import Union, List, Dict, Any
import openai

PAGE_CONFIG = {"page_title":"QuizCrafter", 
               "layout":"centered", 
               "initial_sidebar_state":"auto",
                "page_icon":"🖊️",
               }

st.set_page_config(**PAGE_CONFIG)

together.api_key = 'a0bc508488d11535dca11909486c9c2b8efdd4ede5ab3dce8db507852c23d8e7'

together.Models.start("togethercomputer/llama-2-7b-chat")

class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-7b-chat"
    """model endpoint to use"""

    together_api_key: str = 'a0bc508488d11535dca11909486c9c2b8efdd4ede5ab3dce8db507852c23d8e7'
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text

def strict_output(system_prompt: str,user_prompt: Union[str, List[str]],output_format: Dict[str, Any], default_category: str = "",output_value_only: bool = False,num_tries: int = 1,verbose: bool = False) -> Any:

    list_input = isinstance(user_prompt, list)
    dynamic_elements = any('<' in str(val) and '>' in str(val) for val in output_format.values())
    list_output = any('[' in str(val) and ']' in str(val) for val in output_format.values())
    
    error_msg = ""

    for i in range(num_tries):
        output_format_prompt = f"\nYou are to output {'an array of objects in' if list_output else ''} the following in json format: {json.dumps(output_format)}. \nDo not put quotation marks or escape character \\ in the output fields."

        if list_output:
            output_format_prompt += "\nIf output field is a list, classify output into the best element of the list."

        if dynamic_elements:
            output_format_prompt += "\nAny text enclosed by < and > indicates you must generate content to replace it. Example input: Go to <location>, Example output: Go to the garden\nAny output key containing < and > indicates you must generate the key name to replace it. Example input: {'<location>': 'description of location'}, Example output: {school: a place for education}"

        if list_input:
            output_format_prompt += "\nGenerate an array of json, one json for each input element."

        #p_template = ''' 
        #                {system_prompt} + '\n' + {output_format_prompt} + '\n' + {user_prompt}
#
        #            '''
        #llm = TogetherLLM(
        #        model= "togethercomputer/llama-2-7b-chat",
        #        temperature = 0.7,
        #        max_tokens = 1024
        #    )
        #
        #qa = LLMChain(
        #    llm=ChatOpenAI(temperature=0.7,model='gpt-3.5-turbo',openai_api_key='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'),
        #    prompt= PromptTemplate(template=p_template, input_variables=['system_prompt','output_format_prompt','user_prompt'])
        #)
#
        #res = qa({'system_prompt': system_prompt, 'output_format_prompt': output_value_only, 'user_prompt': str(user_prompt)})['text']
        #print(res)

        openai.api_key ='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'

        conversation = [
            {"role": "system", "content": f"{system_prompt}{output_format_prompt}{error_msg}"},
            {"role": "user", "content": str(user_prompt)}
        ]
        
        # Get OpenAI response
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=conversation,
            temperature=0.7,
        )

        res = response['choices'][0]['message']['content'].replace("'", '"')
        res = res.replace(r"(\w)\"(\w)", r"\1'\2")


        try:
            output = json.loads(res)

            if list_input:
                if not isinstance(output, list):
                    raise ValueError("Output format not in an array of json")
            else:
                output = [output]

            for index in range(len(output)):
                for key in output_format.keys():
                    if "<" in key and ">" in key:
                        continue
                    if key not in output[index]:
                        raise ValueError(f"{key} not in json output")
                    
                    if isinstance(output_format[key], list):
                        choices = output_format[key]
                        if isinstance(output[index][key], list):
                            output[index][key] = output[index][key][0]
                        if output[index][key] not in choices and default_category:
                            output[index][key] = default_category
                        if ":" in output[index][key]:
                            output[index][key] = output[index][key].split(":")[0]

                if output_value_only:
                    output[index] = list(output[index].values())
                    if len(output[index]) == 1:
                        output[index] = output[index][0]

            return output if list_input else output[0]

        except Exception as e:
            error_msg = f"\n\nResult: {res}\n\nError message: {e}"
            print("An exception occurred:", e)
            print('\n\n')
            print("Current invalid json format", res)

    return []
    

def main():

    st.markdown("<h1 style='text-align: center; font-family: Arial;'>🏆</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-family: italic;'><span style='color: #6f42c1;'>Quiz</span>Crafter <span style='background: linear-gradient(to right, #007BFF, #28a745); -webkit-background-clip: text; color: transparent;'>AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; font-family: Courier New;'>Where Algorithms Craft Learning Adventures </h4>", unsafe_allow_html=True)
    st.markdown('<br>' * 2, unsafe_allow_html=True)
    
    if 'res' not in st.session_state:
        st.session_state.res = ''

    with st.form('my_form'):
        user_input = st.text_input("Enter the name of the topic you want to be quizzed about", value = '')
        noqs = st.number_input("Enter the number of questions you want to be quizzed about", value = 4, min_value = 1, max_value = 10)
        choice = st.radio("What kind of questions do you want to be quizzed about?", ('Easy', 'Medium', 'Hard'), key = None, index = 0)
        choice2 = st.radio("What kind of questions do you want to be quizzed about?", ('MCQ', 'Short Answer'), index = 0, key = None)
        submit_button = st.form_submit_button(label='Generate Quiz')
        if submit_button:
            res = strict_output(
                    "You are a helpful AI that is able to generate mcq questions and answers, the length of each answer should not be more than 15 words, store all answers and questions and options in a JSON array. answer should be one among the options. The answer that you generate should be in the options you generate.",
                    [f"You are to generate a random {choice} {choice2} question about {user_input}" for _ in range(noqs)],
                    {
                        "question": "question",
                        "answer": "answer with max length of 15 words",
                        "option1": "option1 with max length of 15 words",
                        "option2": "option2 with max length of 15 words",
                        "option3": "option3 with max length of 15 words",
                        "option4": "option4 with max length of 15 words"
                    }
                )
            st.session_state.res = res
        
    if st.session_state.res != '':
        res = st.session_state.res
        st.markdown('<br>' * 2, unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; font-family: Courier New;'>Your Quiz</h4>", unsafe_allow_html=True)
        with st.form('my_form_2'):
            st.write(f"Question 1: {res[0]['question']}")
            ans1 = st.radio("Choose the correct answer", (res[0]['option1'], res[0]['option2'], res[0]['option3'], res[0]['option4']))
            st.write(f"Question 2: {res[1]['question']}")
            ans2 = st.radio("Choose the correct answer", (res[1]['option1'], res[1]['option2'], res[1]['option3'], res[1]['option4']))
            st.write(f"Question 3: {res[2]['question']}")
            ans3 = st.radio("Choose the correct answer", (res[2]['option1'], res[2]['option2'], res[2]['option3'], res[2]['option4']))
            st.write(f"Question 4: {res[3]['question']}")
            ans4 = st.radio("Choose the correct answer", (res[3]['option1'], res[3]['option2'], res[3]['option3'], res[3]['option4']))
            submit_button2 = st.form_submit_button(label='Submit Answers')

            if submit_button2:
                correct = 0
                if ans1 == res[0]['answer']:
                    correct += 1
                if ans2 == res[1]['answer']:
                    correct += 1
                if ans3 == res[2]['answer']:
                    correct += 1
                if ans4 == res[3]['answer']:
                    correct += 1
                st.write(f"You got {correct} out of {noqs} questions correct!")
                st.write(f"Your score is {correct}/{noqs}")
                st.write(f"Your percentage is {correct/noqs*100}%")
                if correct/noqs*100 >= 90:
                    st.success("You have achieved an A grade!")
                elif correct/noqs*100 >= 80:
                    st.success("You have achieved a B grade!")
                elif correct/noqs*100 >= 70:
                    st.write("You have achieved a C grade!")
                elif correct/noqs*100 >= 60:
                    st.error("You have achieved a D grade!")
                else:
                    st.error("You have achieved a F grade!")    

                st.write("Correct answers:")
                st.write(f"Question 1: {res[0]['answer']}")
                st.write(f"Question 2: {res[1]['answer']}")
                st.write(f"Question 3: {res[2]['answer']}")
                st.write(f"Question 4: {res[3]['answer']}")
                
                st.markdown("<h4 style='text-align: center; font-family: Courier New;'><span style='background: linear-gradient(to right, #8C1AFF, #FF8F00); -webkit-background-clip: text; color: transparent;'> Thank you for using QuizCrafter AI </span></h4>", unsafe_allow_html=True)
                st.balloons()

if __name__ == "__main__":
    main()