# Who Will Cancel? A Hotel Booking Prediction App with Explainable AI and Scenario Simulation
_Predict cancellations. Understand the why. Simulate the what-if._

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![XAI](https://img.shields.io/badge/Explainable%20AI-Enabled-green)

Hotel cancellations don't just affect occupancy — they ripple into revenue, refund policies, marketing strategy, and long-term customer loyalty. This app goes beyond a simple cancellation score by analyzing risk at the room level, not just the booking level. Through strict room-splitting logic, every guest configuration gets its own cancellation profile, making the model's perspective broader and its predictions more grounded in reality. Paired with individual booking explanations and a what-if simulator, hotel teams can move from reacting to cancellations to building smarter SOPs before they happen.

Our goal is straightforward: to quantify how many bookings are truly at risk, predict cancellations with consistent accuracy, evaluate how reliable those predictions are across different booking profiles, and explain the operational factors that drive cancellation behavior.

The project focuses on three core tasks:
1. Identifying which bookings are most likely to cancel — at the room level, not just the booking level,
2. Explaining why each individual booking is flagged as high risk,
3. Simulating what-if scenarios so hotel teams can build smarter policies before cancellations happen.

## How to Run

```bash
git clone https://github.com/username/hotel-cancellation-prediction
cd hotel-cancellation-prediction
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser at `http://localhost:8501`
**Note:** Python 3.10+ recommended.
