# import ollama
import random
from ollama import Client
from ollama import chat, ChatResponse
from typing import Iterator

# from rich import print

# https://deepwiki.com/ollama/ollama-python/4.2-function-calling-with-tools



get_tourist_destinations_tool={
		'type': 'function',
		'function': {
			'name': 'get_tourist_destinations',
			'description': 'Gives a single tourist attraction in a continent',
			'parameters': {
				'type': 'object',
				'required': ['continent'],
				'properties': {
					'continent': {'type': 'string','description': 'name of a continent'},
				},
			},
		},
	}

get_language_time_tool={
		'type': 'function',
		'function': {
			'name': 'get_language_time',
			'description': 'Get the time and language spoken at a location',
			'parameters': {
				'type': 'object',
				'required': ['destination','continent'],
				'properties': {
					'destination': {'type': 'string','description': 'Name of the tourist destination'},
					'continent': {'type': 'string','description': 'Continent of the destination'},
				},
			},
		},
	}

#####################
def get_tourist_destinations(continent:str) -> str:
	"""
	Gives a single tourist attraction in a continent

	Args:
		continent (str): name of a continent

	Returns:
		str: A single tourist destination
	"""
	asia_dest=["Taj Mahal",'Great Wall of China',"Red fort"]
	usa_dest=["Statue of Liberty","Central Park",'Grand Canyon']
	europe_dest=["Buckingham Palace","Northern Lights","Some park"]

	destinations={"Asia":asia_dest,"USA": usa_dest,"Europe":europe_dest}

	return destinations[continent][random.randint(0,len(destinations[continent])-1)]

def get_language_time(destination,continent):
	if continent=="Asia":
		return "Time is morning. Language is Asian."
	else:
		return "Time is Evening. Language is English."


available_functions={
	'get_tourist_destinations':get_tourist_destinations,
	'get_language_time':get_language_time
}

# Think Step-by-step. did not work for executing agent. But great for planning agent.

# TouristInstructions="""
# You are a traveler Agent. You roam to different tourist hotspots and share the following details:
# 1) Continent, country and the city, 2) current time at the location, 3) language spoken by localites and 4) A fact about the destination
# Use the tools provided to you to accomplish your tasks.
# Call the tools sequentially by using output of a tool into the next one.

# The exact steps you have to follow are:
# 1) First you have to pick a continent among Asia, USA, Europe. Pick Asia more often.
# 2) Find top tourist destinations in these continents
# 3) Get details on the destination

# Repeat above steps 3 times. Separte each iteration starting with Iteration number as 
# Iteration 1

# Iteration 2 etc 

# Stop after 3 iterations.
# """


TouristInstructions="""
You are a planning Agent. Create plan to get the time and language of any location using the tools provided to you.
You dont need to execute the plan. Only list the steps you would from the beginning to the end.
Your first step is to pick a continent and it is Europe. Ensure that you use only the tools provided to you and not any internal knowledge.
Think Step-by-Step.
"""
MODEL="qwen3:0.6b"
messages = [{"role": "user", "system":"","content": TouristInstructions}]

client=Client()
finalind=True
while True:
	# response:ChatResponse = chat(model=MODEL, messages=messages,
	# 	tools=[get_tourist_destinations_tool,get_destination_time_tool,get_destination_language_tool], think=True)

	# if response.message.content:
	# 	print('Content: ')
	# 	print(response.message.content + '\n')
	# if response.message.thinking:
	# 	print('Thinking: ')
	# 	print(response.message.thinking + '\n')

	# for part in chat(model=MODEL, messages=messages,
	# 	tools=[get_tourist_destinations_tool,get_destination_time_tool,get_destination_language_tool], 
	# 	think=True,stream=True):
	# 		if part.message.thinking:
	# 			print(part.message.thinking, end='', flush=True)
	response_stream: Iterator[ChatResponse] = client.chat(model=MODEL, messages=messages,
		tools=[get_tourist_destinations_tool,get_language_time_tool], think=True,stream=True)

	tool_calls = []
	thinking = ''
	content = ''

	for chunk in response_stream:
		if chunk.message.tool_calls:
			tool_calls.extend(chunk.message.tool_calls)

		if chunk.message.content:
			if not (chunk.message.thinking or chunk.message.thinking == '') and finalind:
				print('\n\n' + '=' * 10)
				print('Final result: ')
				finalind = False
			print(chunk.message.content, end='', flush=True)

		if chunk.message.thinking:
			# accumulate thinking
			thinking += chunk.message.thinking
			print(chunk.message.thinking, end='', flush=True)

	if thinking != '' or content != '' or len(tool_calls) > 0:
		messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})

	print()

	# messages.append(response.message)

	if tool_calls:
		for tool_call in tool_calls:
			function_to_call = available_functions.get(tool_call.function.name)
			if function_to_call:
				print('\nCalling tool:', tool_call.function.name, 'with arguments: ', tool_call.function.arguments)
				result = function_to_call(**tool_call.function.arguments)
				print('Tool result: ', result + '\n')

				result_message = {'role': 'tool', 'content': result, 'tool_name': tool_call.function.name}
				messages.append(result_message)
			else:
				print(f'Tool {tool_call.function.name} not found')
				messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})

	else:
		# no more tool calls, we can stop the loop
		break