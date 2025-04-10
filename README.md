# Engineer On-Call Scheduling Optimization

## Description

This project implements engineer on-call scheduling optimization models using Google OR-Tools in Python. It aims to generate fair and balanced on-call schedules for a team of engineers over an extended period (e.g., 6 months), considering various constraints, preferences, and balancing objectives.

The project includes multiple scripts:

- **`nurse.py`**: A foundational nurse scheduling model, serving as a basic example.
- **`ops1.py` / `ops2.py`**: Advanced engineer on-call scheduling models with features including:
  - Assigning both a primary **on-call engineer** and a **backup engineer** each day.
  - Distinguishing between weekdays, weekends, and holidays.
  - Handling engineer **vacation requests**.
  - Enforcing **hard constraints** like daily coverage, no double duty (primary and backup on the same day), and minimum rest periods.
  - Optimizing based on **soft constraints** (penalties) to achieve fairness and balance:
    - Maximum total on-call/backup days per engineer.
    - Limits on consecutive on-call/backup duties.
    - Respecting vacation requests.
    - Balancing the number of weekend/holiday assignments among engineers.
    - Balancing the number of assignments per month among engineers.
  - **Different Balancing Strategies:**
    - `ops1.py`: Penalizes deviations from the *average* number of weekend/holiday shifts.
    - `ops2.py`: Uses a *minimax* approach to minimize the *difference* between the maximum and minimum number of weekend/holiday shifts assigned to any engineer.

These scripts define the problem, set up constraints and objective functions (minimizing costs and penalties), solve the optimization problem using the SCIP solver, display the resulting schedule, and provide detailed analysis and visualizations of the schedule's fairness and balance.

## Requirements

The required Python packages are listed in `requirements.txt`. Key dependencies include:

- `ortools`: Google OR-Tools library for optimization.
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and displaying schedules.
- `matplotlib` & `japanize-matplotlib`: For visualizing schedule analysis results.

## Installation

1. **Clone the repository (if applicable):**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the engineer on-call scheduling optimization, execute either `ops1.py` or `ops2.py`:

```bash
# Using the average deviation penalty for weekend/holiday balance
python ops1.py

# Or using the minimax approach for weekend/holiday balance
python ops2.py
```

The scripts will output:

- The solver status and solving time.
- The total objective value and breakdown of costs and penalties (including specific balance penalties).
- The generated on-call and backup schedule, typically displayed month by month.
- Detailed statistics per engineer (total duties, weekend/holiday duties, monthly breakdown).
- Schedule analysis results (load balance, consecutive duties, request compliance).
- Visualizations (plots) summarizing the schedule analysis.

You can also run the basic nurse scheduling example:

```bash
python nurse.py
```

## Configuration

The scheduling parameters (number of engineers, scheduling period, holiday dates, costs, penalty weights, etc.) are defined as constants and variables at the beginning of the `ops1.py` and `ops2.py` scripts. You can modify these values to configure the scheduling problem and tune the balancing objectives.
