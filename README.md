User Request
    │
    ▼
OrchestratorAgent  ──  parses natural language into structured travel params
    │
    ├──► SearchAgent      ──  queries GDS, returns available flights
    ├──► BookingAgent     ──  reserves seats, generates PNR
    ├──► PaymentAgent     ──  charges payment card
    └──► NotificationAgent──  sends email + SMS confirmation
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python agents.py
```

## Customising

### Connecting a real GDS (Amadeus, Sabre, Travelport)
Replace `mock_search_flights()` in agents.py:

```python
import amadeus
client = amadeus.Client(client_id="...", client_secret="...")

def mock_search_flights(origin, destination, date, num_passengers, cabin_class):
    response = client.shopping.flight_offers_search.get(
        originLocationCode=origin,
        destinationLocationCode=destination,
        departureDate=date,
        adults=num_passengers,
        travelClass=cabin_class.upper(),
    )
    return response.data
```

### Real payment (Stripe)
Replace `mock_process_payment()`:

```python
import stripe
stripe.api_key = "sk_live_..."

def mock_process_payment(pnr, amount_usd, card_last4):
    charge = stripe.PaymentIntent.create(
        amount=int(amount_usd * 100),
        currency="usd",
        payment_method_types=["card"],
        metadata={"pnr": pnr},
    )
    return {"charge_id": charge.id, "status": charge.status, "amount_usd": amount_usd}
```

### Real notifications (SendGrid + Twilio)
Replace `mock_send_notification()`:

```python
import sendgrid, twilio.rest

def mock_send_notification(pnr, email, phone, booking_details):
    # SendGrid email
    sg = sendgrid.SendGridAPIClient(api_key="SG....")
    sg.send(Mail(from_email="noreply@airline.com", to_emails=email, ...))

    # Twilio SMS
    tw = twilio.rest.Client("AC...", "auth_token")
    tw.messages.create(body=f"Confirmed PNR {pnr}", from_="+1...", to=phone)
    return {"email_sent": True, "sms_sent": True}
```

## Extending with more agents

Add a `PricingAgent` for dynamic fare rules, a `LoyaltyAgent` to apply miles,
or a `ChangeAgent` to handle date modifications post-booking.
Each follows the same pattern: subclass `BaseAgent`, define tools, implement a
`handler` function, call `self.run(prompt, handler)`.
