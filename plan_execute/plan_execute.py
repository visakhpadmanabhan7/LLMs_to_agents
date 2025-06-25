from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_chat_planner,
    load_agent_executor,
)

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

def find_flight(query: str) -> str:
    if "New York to Paris" in query:
        return "Flight booked: NY to Paris on July 20, departs 8PM, arrives 9AM (local). Cost: $650"
    return "No flights found."

def book_hotel(query: str) -> str:
    if "check-in" in query.lower():
        return "Hotel booked: Paris Inn from July 20, 10AM for 3 nights. Cost: $400"
    return "No suitable hotels."

def arrange_transfer(query: str) -> str:
    if "from airport" in query:
        return "Transfer arranged: Pickup at 9:30AM, Cost: $50"
    return "Could not schedule transfer."

tools = [
    Tool(name="FlightSearch", func=find_flight, description="Search and book flights between cities"),
    Tool(name="HotelBooking", func=book_hotel, description="Book a hotel given check-in time and location"),
    Tool(name="TransferBooking", func=arrange_transfer, description="Book airport to hotel transfer"),
]

planner = load_chat_planner(llm)
executor = load_agent_executor(llm=llm, tools=tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

task = (
    "Book a flight from New York to Paris on July 20 for under $700, "
    "then book a hotel that matches the flight arrival, "
    "and finally arrange an airport pickup. "
    "Total budget is $1200."
)

result = agent.invoke({"input": task})
print(result["output"])