"""
Flight Booking Agent System
============================
Multi-agent system using the Anthropic API (tool use) to search,
book, charge, and notify for flight reservations.

Agents:
  OrchestratorAgent   - parses user intent, routes to sub-agents
  SearchAgent         - queries mock GDS for available flights
  BookingAgent        - reserves seats and creates PNR
  PaymentAgent        - processes card charge
  NotificationAgent   - sends email/SMS confirmation

Run:
  python agents.py
"""

import anthropic
import json
import uuid
import random
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Shared booking state
# ---------------------------------------------------------------------------

@dataclass
class PassengerInfo:
    name: str
    passport: str
    dob: str  # YYYY-MM-DD


@dataclass
class BookingState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    num_passengers: int = 1
    cabin_class: str = "economy"
    passengers: list[PassengerInfo] = field(default_factory=list)
    selected_flight: Optional[dict] = None
    pnr: Optional[str] = None
    payment_status: Optional[str] = None
    total_charged: Optional[float] = None
    confirmation_sent: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock external services  (replace with real APIs in production)
# ---------------------------------------------------------------------------

MOCK_FLIGHTS_DB = [
    {"flight_id": "AI101", "airline": "Air India",     "origin": "DEL", "destination": "JFK",
     "departure": "08:00", "arrival": "14:30+1", "duration": "14h 30m",
     "price_usd": 780, "seats_available": 12, "stops": 0},
    {"flight_id": "UA789", "airline": "United",        "origin": "DEL", "destination": "JFK",
     "departure": "23:45", "arrival": "06:10+2", "duration": "16h 25m",
     "price_usd": 640, "seats_available": 5,  "stops": 1},
    {"flight_id": "EK212", "airline": "Emirates",      "origin": "DEL", "destination": "JFK",
     "departure": "03:35", "arrival": "11:55+1", "duration": "16h 20m",
     "price_usd": 920, "seats_available": 8,  "stops": 1},
    {"flight_id": "LH756", "airline": "Lufthansa",     "origin": "DEL", "destination": "JFK",
     "departure": "14:15", "arrival": "23:45+1", "duration": "17h 30m",
     "price_usd": 710, "seats_available": 20, "stops": 1},
]


def mock_search_flights(origin: str, destination: str, date: str,
                        num_passengers: int, cabin_class: str) -> list[dict]:
    """Simulate a GDS flight search."""
    multiplier = {"economy": 1.0, "business": 3.2, "first": 5.5}.get(cabin_class, 1.0)
    results = []
    for f in MOCK_FLIGHTS_DB:
        if f["origin"].upper() == origin.upper() and \
           f["destination"].upper() == destination.upper() and \
           f["seats_available"] >= num_passengers:
            flight = f.copy()
            flight["price_usd"] = round(f["price_usd"] * multiplier * num_passengers, 2)
            flight["date"] = date
            results.append(flight)
    return results


def mock_create_booking(flight_id: str, date: str,
                        passengers: list[dict], cabin_class: str) -> dict:
    """Simulate inventory reservation and PNR creation."""
    pnr = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
    return {
        "pnr": pnr,
        "flight_id": flight_id,
        "date": date,
        "cabin_class": cabin_class,
        "passengers": passengers,
        "status": "confirmed",
        "booked_at": datetime.now(timezone.utc).isoformat(),
    }


def mock_process_payment(pnr: str, amount_usd: float,
                         card_last4: str = "4242") -> dict:
    """Simulate a payment gateway charge."""
    charge_id = "ch_" + uuid.uuid4().hex[:16]
    return {
        "charge_id": charge_id,
        "pnr": pnr,
        "amount_usd": amount_usd,
        "card_last4": card_last4,
        "status": "succeeded",
        "charged_at": datetime.now(timezone.utc).isoformat(),
    }


def mock_send_notification(pnr: str, email: str, phone: str,
                           booking_details: dict) -> dict:
    """Simulate email + SMS dispatch."""
    print(f"\n  [EMAIL] -> {email}")
    print(f"  Subject: Booking Confirmed - PNR {pnr}")
    print(f"  Flight {booking_details['flight_id']} on {booking_details['date']}")
    print(f"  Passengers: {len(booking_details['passengers'])}")
    print(f"\n  [SMS] -> {phone}: Your flight {booking_details['flight_id']} is confirmed. PNR: {pnr}")
    return {"email_sent": True, "sms_sent": True, "pnr": pnr}


# ---------------------------------------------------------------------------
# Tool definitions for each agent
# ---------------------------------------------------------------------------

SEARCH_TOOLS = [
    {
        "name": "search_flights",
        "description": "Search available flights between two airports on a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin":        {"type": "string", "description": "IATA code, e.g. DEL"},
                "destination":   {"type": "string", "description": "IATA code, e.g. JFK"},
                "date":          {"type": "string", "description": "YYYY-MM-DD"},
                "num_passengers":{"type": "integer", "description": "Number of passengers"},
                "cabin_class":   {"type": "string", "enum": ["economy", "business", "first"]},
            },
            "required": ["origin", "destination", "date", "num_passengers", "cabin_class"],
        },
    }
]

BOOKING_TOOLS = [
    {
        "name": "create_booking",
        "description": "Reserve seats for a flight and obtain a PNR (booking reference).",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id":   {"type": "string"},
                "date":        {"type": "string", "description": "YYYY-MM-DD"},
                "cabin_class": {"type": "string", "enum": ["economy", "business", "first"]},
                "passengers":  {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":     {"type": "string"},
                            "passport": {"type": "string"},
                            "dob":      {"type": "string"},
                        },
                        "required": ["name", "passport", "dob"],
                    },
                },
            },
            "required": ["flight_id", "date", "cabin_class", "passengers"],
        },
    }
]

PAYMENT_TOOLS = [
    {
        "name": "process_payment",
        "description": "Charge the customer's card for a confirmed booking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pnr":        {"type": "string", "description": "Booking reference"},
                "amount_usd": {"type": "number"},
                "card_last4": {"type": "string", "description": "Last 4 digits of card"},
            },
            "required": ["pnr", "amount_usd", "card_last4"],
        },
    }
]

NOTIFICATION_TOOLS = [
    {
        "name": "send_notification",
        "description": "Send email and SMS booking confirmation to the passenger.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pnr":             {"type": "string"},
                "email":           {"type": "string"},
                "phone":           {"type": "string"},
                "booking_details": {"type": "object"},
            },
            "required": ["pnr", "email", "phone", "booking_details"],
        },
    }
]

ORCHESTRATOR_TOOLS = [
    {
        "name": "parse_travel_request",
        "description": "Extract structured travel parameters from a natural language request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin":           {"type": "string", "description": "IATA airport code"},
                "destination":      {"type": "string", "description": "IATA airport code"},
                "departure_date":   {"type": "string", "description": "YYYY-MM-DD"},
                "return_date":      {"type": "string", "description": "YYYY-MM-DD or null for one-way"},
                "num_passengers":   {"type": "integer"},
                "cabin_class":      {"type": "string", "enum": ["economy", "business", "first"]},
            },
            "required": ["origin", "destination", "departure_date", "num_passengers", "cabin_class"],
        },
    }
]


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list, model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
        self.client = anthropic.Anthropic()

    def run(self, user_message: str, tool_handler) -> str:
        """Run an agentic loop until the model stops requesting tools."""
        messages = [{"role": "user", "content": user_message}]
        print(f"\n{'-'*60}")
        print(f"  Agent: {self.name}")
        print(f"  Input: {user_message[:120]}...")
        print(f"{'-'*60}")

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            # Collect text output
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            if text_parts:
                print(f"  [{self.name}]: {' '.join(text_parts)[:200]}")

            if response.stop_reason == "end_turn":
                return " ".join(text_parts)

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  -> Tool call: {block.name}({json.dumps(block.input)[:120]})")
                    result = tool_handler(block.name, block.input)
                    print(f"  <- Result: {json.dumps(result)[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            # Append assistant message + tool results and loop
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            system_prompt="""You are the orchestrator for a flight booking system.
Your job is to parse the user's travel request using the parse_travel_request tool
and extract all required fields. Be precise with IATA airport codes.
If dates are relative (e.g. "next Friday"), convert them to YYYY-MM-DD format.
Today's date for reference: """ + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            tools=ORCHESTRATOR_TOOLS,
        )
        self.parsed: dict = {}

    def parse_request(self, user_input: str) -> dict:
        def handler(tool_name, tool_input):
            if tool_name == "parse_travel_request":
                self.parsed = tool_input
                return {"status": "parsed", "data": tool_input}
            return {"error": "unknown tool"}

        self.run(user_input, handler)
        return self.parsed


class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SearchAgent",
            system_prompt="""You are a flight search agent. Use the search_flights tool
to find available options. Return a clear list of flights with prices, times, and airlines.
Always call the tool — never invent flight data.""",
            tools=SEARCH_TOOLS,
        )
        self.results: list = []

    def search(self, origin: str, destination: str, date: str,
               num_passengers: int, cabin_class: str) -> list[dict]:
        def handler(tool_name, tool_input):
            if tool_name == "search_flights":
                flights = mock_search_flights(**tool_input)
                self.results = flights
                return {"flights": flights, "count": len(flights)}
            return {"error": "unknown tool"}

        prompt = (f"Search flights from {origin} to {destination} on {date} "
                  f"for {num_passengers} passenger(s) in {cabin_class} class.")
        self.run(prompt, handler)
        return self.results


class BookingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="BookingAgent",
            system_prompt="""You are a flight booking agent. Use the create_booking tool
to reserve seats once a flight has been selected. Confirm all passenger details are complete
before calling the tool.""",
            tools=BOOKING_TOOLS,
        )
        self.booking_result: dict = {}

    def book(self, flight: dict, passengers: list[dict], cabin_class: str) -> dict:
        def handler(tool_name, tool_input):
            if tool_name == "create_booking":
                result = mock_create_booking(**tool_input)
                self.booking_result = result
                return result
            return {"error": "unknown tool"}

        prompt = (f"Book flight {flight['flight_id']} on {flight['date']} "
                  f"in {cabin_class} class for passengers: {json.dumps(passengers)}")
        self.run(prompt, handler)
        return self.booking_result


class PaymentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PaymentAgent",
            system_prompt="""You are a payment processing agent. Use the process_payment tool
to charge the customer. Verify the PNR and amount before processing.
Never process a payment without a valid PNR.""",
            tools=PAYMENT_TOOLS,
        )
        self.payment_result: dict = {}

    def charge(self, pnr: str, amount_usd: float, card_last4: str = "4242") -> dict:
        def handler(tool_name, tool_input):
            if tool_name == "process_payment":
                result = mock_process_payment(**tool_input)
                self.payment_result = result
                return result
            return {"error": "unknown tool"}

        prompt = (f"Process payment for PNR {pnr}. "
                  f"Amount: USD {amount_usd:.2f}. Card ending: {card_last4}.")
        self.run(prompt, handler)
        return self.payment_result


class NotificationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="NotificationAgent",
            system_prompt="""You are a notification agent. Use the send_notification tool
to dispatch a booking confirmation email and SMS to the passenger.
Always include the PNR and flight details in the notification.""",
            tools=NOTIFICATION_TOOLS,
        )

    def notify(self, pnr: str, email: str, phone: str, booking_details: dict) -> dict:
        result = {}

        def handler(tool_name, tool_input):
            nonlocal result
            if tool_name == "send_notification":
                result = mock_send_notification(**tool_input)
                return result
            return {"error": "unknown tool"}

        prompt = (f"Send booking confirmation for PNR {pnr} to {email} / {phone}. "
                  f"Booking details: {json.dumps(booking_details)}")
        self.run(prompt, handler)
        return result


# ---------------------------------------------------------------------------
# Main orchestration pipeline
# ---------------------------------------------------------------------------

def run_booking_pipeline(user_request: str, passenger_details: list[dict],
                          contact_email: str, contact_phone: str,
                          card_last4: str = "4242") -> BookingState:
    """
    End-to-end flight booking pipeline.

    Args:
        user_request:      Natural language request, e.g. "Book DEL to JFK on 2026-04-25 for 1 adult economy"
        passenger_details: List of dicts with keys: name, passport, dob
        contact_email:     Email for confirmation
        contact_phone:     Phone for SMS
        card_last4:        Last 4 digits of payment card

    Returns:
        BookingState with all results populated
    """
    state = BookingState()
    print(f"\n{'='*60}")
    print(f"  FLIGHT BOOKING SYSTEM  |  Session: {state.session_id}")
    print(f"{'='*60}")
    print(f"  Request: {user_request}")

    # -- Step 1: Parse intent ----------------------------------------------
    print("\n[STEP 1] Parsing travel request...")
    orchestrator = OrchestratorAgent()
    parsed = orchestrator.parse_request(user_request)

    if not parsed:
        state.errors.append("Failed to parse travel request")
        return state

    state.origin          = parsed.get("origin", "").upper()
    state.destination     = parsed.get("destination", "").upper()
    state.departure_date  = parsed.get("departure_date")
    state.return_date     = parsed.get("return_date")
    state.num_passengers  = parsed.get("num_passengers", 1)
    state.cabin_class     = parsed.get("cabin_class", "economy")

    print(f"\n  Parsed: {state.origin} -> {state.destination} | "
          f"{state.departure_date} | {state.num_passengers} pax | {state.cabin_class}")

    # -- Step 2: Search flights --------------------------------------------
    print("\n[STEP 2] Searching flights...")
    search_agent = SearchAgent()
    flights = search_agent.search(
        state.origin, state.destination,
        state.departure_date, state.num_passengers, state.cabin_class,
    )

    if not flights:
        state.errors.append(f"No flights found from {state.origin} to {state.destination} on {state.departure_date}")
        print("  No flights found.")
        return state

    print(f"\n  Found {len(flights)} flight(s):")
    for i, f in enumerate(flights, 1):
        print(f"  [{i}] {f['airline']} {f['flight_id']} | {f['departure']}->{f['arrival']} "
              f"| {f['duration']} | USD {f['price_usd']} | {f['stops']} stop(s)")

    # Auto-select cheapest flight (in production: present to user)
    state.selected_flight = min(flights, key=lambda x: x["price_usd"])
    print(f"\n  Auto-selected: {state.selected_flight['airline']} {state.selected_flight['flight_id']} "
          f"(USD {state.selected_flight['price_usd']})")

    # -- Step 3: Create booking --------------------------------------------
    print("\n[STEP 3] Creating booking...")
    booking_agent = BookingAgent()
    booking = booking_agent.book(
        state.selected_flight, passenger_details, state.cabin_class
    )

    if not booking.get("pnr"):
        state.errors.append("Booking creation failed")
        return state

    state.pnr = booking["pnr"]
    print(f"\n  PNR created: {state.pnr}")

    # -- Step 4: Process payment -------------------------------------------
    print("\n[STEP 4] Processing payment...")
    payment_agent = PaymentAgent()
    payment = payment_agent.charge(
        state.pnr, state.selected_flight["price_usd"], card_last4
    )

    state.payment_status = payment.get("status", "failed")
    state.total_charged  = payment.get("amount_usd")

    if state.payment_status != "succeeded":
        state.errors.append(f"Payment failed: {payment}")
        return state

    print(f"\n  Payment succeeded | Charge ID: {payment.get('charge_id')} | "
          f"USD {state.total_charged}")

    # -- Step 5: Send notifications ----------------------------------------
    print("\n[STEP 5] Sending confirmation...")
    notify_agent = NotificationAgent()
    notify_agent.notify(
        pnr=state.pnr,
        email=contact_email,
        phone=contact_phone,
        booking_details={
            "flight_id": state.selected_flight["flight_id"],
            "airline":   state.selected_flight["airline"],
            "date":      state.departure_date,
            "departure": state.selected_flight["departure"],
            "arrival":   state.selected_flight["arrival"],
            "cabin":     state.cabin_class,
            "passengers": passenger_details,
            "amount_usd": state.total_charged,
        },
    )
    state.confirmation_sent = True

    # -- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  BOOKING COMPLETE")
    print(f"  PNR:        {state.pnr}")
    print(f"  Flight:     {state.selected_flight['airline']} {state.selected_flight['flight_id']}")
    print(f"  Route:      {state.origin} -> {state.destination}")
    print(f"  Date:       {state.departure_date}")
    print(f"  Cabin:      {state.cabin_class.title()}")
    print(f"  Passengers: {state.num_passengers}")
    print(f"  Charged:    USD {state.total_charged}")
    print(f"  Notified:   {contact_email} / {contact_phone}")
    print(f"{'='*60}\n")

    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    final_state = run_booking_pipeline(
        user_request="I need to book a flight from Delhi to New York JFK on April 25th 2026 for 1 adult in economy class",
        passenger_details=[
            {"name": "Arjun Sharma", "passport": "P1234567", "dob": "1990-03-15"},
        ],
        contact_email="arjun.sharma@example.com",
        contact_phone="+91-9876543210",
        card_last4="4242",
    )

    if final_state.errors:
        print("Errors encountered:")
        for e in final_state.errors:
            print(f"  - {e}")