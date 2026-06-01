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

# Understanding Our Deliveries
Each row in this dataset tells the story of one hotel booking:
- Who is staying? — number of adults, children, and babies
- Where are they staying? — which hotel
- When are they coming? — arrival date, how far in advance they booked
- How are they booking? — through which channel, what deposit type
- Did they cancel? — that's what we're trying to predict

Before any scores, we clean and make sense of every column so the results remain trustworthy.

# Understanding Our Bookings

Each row in this dataset tells the story of one hotel booking:
- **Who** is staying? — number of adults, children, and babies
- **Where** are they staying? — which hotel, which country
- **When** are they coming? — arrival date, how far in advance they booked
- **How** are they booking? — through which channel, what deposit type
- **Did they cancel?** — that's what we're trying to predict

Before any machine learning, we clean and make sense of every column so the results remain trustworthy.

| Column | Type | What It Means | Notes & Handling |
|--------|------|---------------|-----------------|
| `bookingID` | `int` | Unique identifier for each booking | Used for room-level splitting; **dropped before modeling** so the model does not memorise IDs |
| `hotel` | `category` | Which hotel the guest booked | 64 unique hotels; cleaned and encoded |
| `is_canceled` | `int (0/1)` | Did the guest cancel? | **Target variable** — this is what the model predicts |
| `lead_time` | `int` | How many days before arrival the booking was made | Longer lead time = higher cancel risk; avg 144 days for cancellations vs 80 days for non-cancellations |
| `arrival_date_year` | `int` | Year of arrival | Used for train/test time-based split |
| `arrival_date_month` | `category` | Month of arrival | Cleaned; used to engineer `season` feature |
| `arrival_date_week_number` | `int` | Week number of arrival | Supports seasonal pattern detection |
| `arrival_date_day_of_month` | `int` | Day of month of arrival | Granular arrival timing |
| `stays_in_weekend_nights` | `int` | Number of weekend nights booked | Combined with week nights to get total stay length |
| `stays_in_week_nights` | `int` | Number of weekday nights booked | Combined with weekend nights to get total stay length |
| `adults` | `int` | Number of adults in the booking | Used in room-splitting logic — max 4 guests per room |
| `children` | `float` | Number of children in the booking | 3 missing values → imputed with median; must have at least 1 adult per room |
| `babies` | `int` | Number of babies in the booking | Treated same as children in room-splitting rules |
| `meal` | `category` | Meal plan selected (BB, HB, FB, SC, Undefined) | Cleaned and encoded |
| `country` | `category` | Guest's country of origin | 346 missing values → imputed with most frequent; 165 unique countries |
| `market_segment` | `category` | How the guest found the hotel (e.g. Online TA, Direct, Groups) | Groups cancel at 60%; Online TA at 37% |
| `distribution_channel` | `category` | Channel through which booking was made (e.g. TA/TO, Direct) | Cleaned and encoded |
| `is_repeated_guest` | `int (0/1)` | Has this guest stayed before? | Repeated guests cancel only 10% of the time vs 38% for new guests |
| `previous_cancellations` | `int` | How many times this guest cancelled in the past | Guests with prior cancellations cancel again 91% of the time |
| `previous_bookings_not_canceled` | `int` | How many past bookings this guest completed | Higher count = more trustworthy guest |
| `reserved_room_type` | `category` | Room type the guest originally requested | Compared with assigned room to detect mismatches |
| `assigned_room_type` | `category` | Room type the guest was actually given | **Dropped before modeling** — only known after check-in, would cause data leakage |
| `booking_changes` | `int` | How many times the booking was modified | Changes may indicate engaged or uncertain guests |
| `deposit_type` | `category` | Type of deposit paid (No Deposit, Non Refund, Refundable) | Non Refund bookings cancel 99.3% of the time — a key signal |
| `agent` | `float` | ID of the travel agent who made the booking | 11,404 missing values → imputed with median; used to detect bulk-booking agents |
| `company` | `float` | ID of the company who made the booking | 78,559 missing values (94%) → **dropped** due to excessive missingness |
| `days_in_waiting_list` | `int` | How many days the booking sat on the waiting list | Waiting list bookings may behave differently |
| `customer_type` | `category` | Type of customer (Transient, Contract, Group, Transient-Party) | Transient customers cancel at 41% — the highest among all types |
| `adr` | `float` | Average daily rate — the room price per night | Used to estimate revenue at risk; engineered into `room_revenue` and `booking_revenue` |
| `required_car_parking_spaces` | `int` | Number of parking spaces requested | Guests who request parking cancel less often |
| `total_of_special_requests` | `int` | Number of special requests made | Guests with zero requests cancel at 47% vs 22% for those with requests |
| `reservation_status` | `category` | Final status of the booking (Check-Out, Canceled, No-Show) | **Dropped before modeling** — this is a post-event label, direct leakage of the target |
| `reservation_status_date` | `date` | Date the reservation status was last updated | Used only for train/test split boundary; **dropped before modeling** |

**Before any model runs, the data goes through a cleaning pass:**
- Extra spaces and typos in text columns are fixed automatically
- Missing values are filled — numbers use the median, categories use the most common value
- Columns like company (94% missing) are dropped entirely
- Columns that would "cheat" the model — like reservation_status and assigned_room_type — are removed before training
- Data is split 80% for training, 20% for testing, using a time-based boundary so the model never sees future data during training

# Attachment
- [Data Processing](https://colab.research.google.com/drive/1_NC888eTnuIVT-aEAOw0Sttztr5aiPS_?usp=share_link)

# What We Bring To The Table? 

## Why the Model Is Useful?

### What are we trying to answer here?
- **Hotels confirm too easily:** bookings are accepted and locked in without clear conditions, especially for bulk reservations. No deposit tiers, no cancellation windows, just "confirmed."
- **No memory of guest behavior:** a guest who cancelled three times before walks in with the same treatment as a first-time loyal customer. The system does not remember, so the pattern repeats.
- **Room for manipulation:** certain booking patterns suggest fictitious reservations, this appears concentrated in specific marketing channels.

### What are we trying to answer here?
- **Hotels often confirm bookings without conditions:** no deposit tiers, no cancellation windows, no memory of past guest behavior. The result is a system that is easy to game and hard to protect.
- **A 37% cancellation rate does not mean 37% of guests are unreliable.** It means the booking process itself has gaps: bulk bookings with no deposit requirements, repeat cancellers treated like first-time guests, and marketing channels with suspicious booking patterns.
- **The real question is not "will this booking cancel?"** - it is "is this booking serious enough to hold a room for?"
- This model exists to answer that question at the moment it matters most: **when the booking comes in, not after the guest fails to show up.**

### How do we approach this?
- **The data starts at the booking level**, but a booking for 5 people (especially when it involve minors in between) **sometimes is not one room, it is two**. So the first step is splitting every booking into individual rooms, because cancellation risk lives at the room level, not just the booking level.
- From there, **new columns are made engineered-ly to surface patterns the raw data hides:** how long until arrival, how much revenue is at risk per room, whether the guest has any history of cancelling, and whether the booking came through a channel that tends to cancel more.
- **Those patterns are then confirmed:** certain months, certain hotels, certain deposit types, certain market segments all show statistically different cancel rates. The data is not guessing; it is showing.
- **The cleaned, enriched dataset is split 80/20:** 80% to train the model, 20% to test whether it holds up on bookings it has never seen.
- **The output is a cancellation probability per booking** and from there, the app lets you drill down: filter by segment, pick a specific booking, and see exactly which factors made that guest a risk.

### What do we actually get out of it?
A hotel manager who opens this app every morning does not just see numbers — they see decisions waiting to be made.
- **For operations:** rooms are no longer prepared blindly. High-risk bookings are flagged in advance, so the team knows which reservations need follow-up and which ones are solid enough to prepare for.
- **For marketing:** bonus structures can shift from rewarding bookings confirmed to bookings completed — closing the gap that allows fictitious or low-commitment reservations to slip through.
- **For customer policy:** deposit tiers now have a data foundation. Instead of a blanket "no deposit" or "non-refund" rule, the app shows which booking profiles genuinely need a deposit to be taken seriously — and which ones are low-risk enough to stay flexible.
- **The bottom line:** the app does not replace human judgment, it gives that judgment something solid to stand on. Every SOP decision, from refund windows to bulk booking requirements, can now point to data instead of gut feeling.

## How Big the Problem Is?

### What are we trying to understand about the data?
- A cancellation is not just a lost booking. It is a preparation that never needed to happen, a room that sat empty during peak season, a marketing promo built on a reservation that was never serious, and an operations team that spent the day getting ready for guests who never came.
- When this happens repeatedly — and the data shows it does, **37% of all bookings cancel** — the damage compounds. Staff work without clear goals. Schedules are built around bookings that dissolve. And because there was never a warning system, the same guests cancel again and again with no consequence.
- **The dataset itself reflects this chaos:** bookings are recorded at the reservation level, not the room level — so nobody actually knows _how many rooms were sold_, _how many were genuinely occupied_, and _how many were just numbers on a spreadsheet_.
- This section exists to make that problem visible: in numbers, in patterns, and in the segments where it hurts the most, **before the model tries to fix it**.

### What does the data actually show?
- The first thing the data reveals is a question: "What was this system actually optimising for? **The number of bookings**, or **the number of rooms genuinely filled?**" Because the two are very different things, and the data suggests only one was being tracked.
- Bookings are recorded by ID with no room-level breakdown. A reservation for 13 rooms packed into a reservation for 1. **Nobody in the system was asking how many beds were actually being prepared.**
- Once you look past that, the patterns are not surprising — they are inevitable:

| What the data shows | The number |
|---------------------|------------|
| Overall cancellation rate | **37%** — nearly 1 in 3 bookings |
| Guests who cancelled before, cancelling again | **91.3%** — and they were let back in every time |
| Non Refund bookings that still cancelled | **99.3%** — the deposit type meant nothing |
| Group bookings that cancelled | **60.7%** — the largest and riskiest segment |
| Bookings with no special requests that cancelled | **47.6%** — no engagement, no commitment |
| Average lead time for cancellations | **144 days** — booked far ahead, cancelled without consequence |

<table align="center">
  <tr>
    <td><img src="https://github.com/Catherinerezi/Cancel-Prediction-Prosperity/blob/main/assets/Chart%20cancel%20rate%20per%20deposit%20type.png" alt="Cancel rate by deposit type" width="450"></td>
    <td><img src="https://github.com/Catherinerezi/Cancel-Prediction-Prosperity/blob/main/assets/Chart%20cancel%20rate%20per%20market%20segment.png" alt="Cancel rate by market segment" width="450"></td>
  </tr>
</table>

- A guest **who has cancelled multiple times before walks in with the same welcome as a first-time loyal customer**. A group booking 13 rooms with no deposit, made 5 months in advance, gets confirmed like any other. Peak season or not.
- The data does not reveal a mystery. _It reveals a system that was never designed to say no._

### How do we explore this in practice?
- The EDA section of the app is built around one simple principle: the further right the bar, the worse the cancellation problem in that category.
- Instead of reading a static report, hotel teams can filter interactively — by deposit type, market segment, customer type, arrival month, and more — and watch the cancel rates shift in real time.
- But the app also surfaces the findings that matter most, so you do not have to go looking:
  - **Non Refund and No Deposit bookings sit at the top of the cancel chart:** bookings with no financial commitment cancel at the highest rates
  - **Group market segment is the most dangerous bulk category:** cancelling at over 60%, second only to Undefined which represents unknown or mixed sources
  - **Lead time is a consistent signal:** cancellations cluster heavily at 144+ days before arrival, meaning the further out a booking is made, the less it should be trusted without a deposit
  - **Peak season is not protected:** a disproportionate share of cancellations happen in the lead-up to high-demand periods, exactly when an empty room hurts the most

### What picture does this paint?
- The data does not just show a cancellation problem — it shows **a system without boundaries**. No clear rules for bulk bookings, no consequences for repeat cancellers, no deposit structure that means anything. The result is **a spreadsheet full of numbers that look like revenue but are not**.
- What this data is really asking for **is not a better algorithm.** _It is a clearer SOP_ — one that applies to customers just as much as it applies to staff. When the rules are clear, operations know who they are preparing for, marketing has no room to game the system, and the guests who do show up are the ones who were serious from the start.
- **A hotel with honest data is a hotel that can plan.** And honest data starts with bookings that mean something.

## What Makes This Approach Different?

### What are we trying to answer here?
- **The raw data treats every booking the same:** one row, one ID, regardless of whether it is 1 room or 13. That means bulk bookings, which cancel at over 60%, are invisible in the data unless you go looking for them.
- But there is a second problem: **the guest composition is not realistic.** _Children and infants appear in bookings with no adult assigned to their room_ — which does not happen in practice. A baby does not check in alone.
- So the question this section answers is: "If we split every booking into individual rooms the way they would actually be assigned — with realistic occupancy rules — does the cancellation picture change?"
- It does. And that difference is what makes this model's view of bulk booking risk more grounded than a standard per-booking prediction.

### How do we approach this?
- Every booking is broken down into individual rooms using a strict set of occupancy rules that reflect how hotels actually work — not just how the data is recorded.
- The rules are straightforward:
  - Maximum 4 guests per room — adults, children, and infants combined, assuming standard room type with no extra charge for additional beds
  - Every room with a child or infant must have at least 1 adult — a child cannot check in alone, so the system always assigns an adult first before filling the remaining 3 spots with minors
  - Leftover adults fill remaining rooms — after all children and infants are placed with an adult, any remaining adults are grouped together up to 4 per room
  - No child or infant is ever left without an adult — if after splitting there is 1 child remaining, they are merged into a room that still has capacity rather than placed alone
- The result is a room-level dataset where every row represents a realistic, occupiable room — not just a line in a booking system.
- Bulk bookings of 3 or more rooms are automatically flagged, because that threshold is where cancellation risk jumps significantly.
