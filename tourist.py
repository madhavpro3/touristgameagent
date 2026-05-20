# import ollama
import random
from ollama import chat

TouristInstructions="""
You are a traveler Agent. You roam to different tourist hotspots and share the following details:
1) Continent, country and the city, 2) current time at the location, 3) language spoken by localites and 4) A fact about the destination
Use the tools provided to you to accomplish your tasks.

The exact steps you have to follow are:
1) First you have to pick a continent among Asia, USA, Europe
2) Find top tourist destinations in these continents
3) Get details on the destination
"""
# 4) Repeat steps 1-3 3 times


def get_tourist_destinations(continent):
	"""
	Gives 1 tourist attraction in a continent

	Args:
	continent: name of a continent

	Returns:
	A single tourist destination
	"""
	asia_dest=["Taj Mahal",'Great Wall of China',"Red fort"]
	usa_dest=["Statue of Liberty","Central Park",'Grand Canyon']
	europe_dest=["Buckingham Palace","Northern Lights","Some park"]

	destinations={"Asia":asia_dest,"USA": usa_dest,"Europe":europe_dest}

	return destinations[continent][random.randint(0,len(destinations[continent])-1)]

def get_destination_time(destination,continent):
	"""
	Get the time at a location

	Args:
	destination: Name of the tourist destination
	continent: Continent of the destination

	Returns:
	Time at the location
	"""
	if continent=="Asia":
		return "Morning"
	else:
		return "Afternoon/Night"

def get_destination_language(destination,continent):
	"""
	Gives the language spoken at the location

	Args:
	destination: Name of the tourist destination
	continent: Continent of the destination

	Returns:
	Language spoken by localites at the destination
	"""
	if continent=="Asia":
		return "Asian"
	else:
		return "English"


MODEL="qwen3:0.6b"
messages = [{"role": "user", "system":"","content": TouristInstructions}]

# pass functions directly as tools in the tools list or as a JSON schema
response = chat(model=MODEL, messages=messages,
	tools=[get_tourist_destinations,get_destination_time,get_destination_language], think=True)
# get_tourist_destinations,get_destination_time,get_destination_language
print(response.message.thinking)

messages.append(response.message)
print(response.message.tool_calls)
# while response.message.tool_calls:
#   # only recommended for models which only return a single tool call
#   call = response.message.tool_calls[0]
#   result = get_direction(**call.function.arguments)
#   # add the tool result to the messages
#   messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

#   final_response = chat(model=MODEL, messages=messages,
# 	tools=[get_tourist_destinations,get_destination_time,get_destination_language], think=True)
#   # final_response = chat(model=MODEL, messages=messages, think=False)
#   print(response.message.thinking)
#   print(final_response.message.content)