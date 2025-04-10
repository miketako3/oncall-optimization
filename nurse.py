from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import time

# 問題の定数
NUM_NURSES = 10  # 看護師の数
NUM_DAYS = 14    # スケジューリング期間（日数）
NUM_SHIFTS = 4   # シフトの種類: 0:休み, 1:早番, 2:遅番, 3:夜勤

# シフトの名称
SHIFT_NAMES = ["休み", "早番", "遅番", "夜勤"]

# パラメータ
# 各シフトの需要 (各日、各シフトの最低必要人数）
demands = np.zeros((NUM_DAYS, NUM_SHIFTS), dtype=int)
# 日勤（早番と遅番）に必要な看護師は各3人、夜勤に必要な看護師は2人
demands[:, 1] = 3  # 早番
demands[:, 2] = 3  # 遅番
demands[:, 3] = 2  # 夜勤

# 各看護師の最大労働日数
max_shifts = np.full(NUM_NURSES, 10)  # 各看護師は最大10日まで勤務可能

# 各看護師の希望シフト (1: 希望する, 0: 希望しない)
# 簡単のためランダムに生成
np.random.seed(42)
requests = np.zeros((NUM_NURSES, NUM_DAYS, NUM_SHIFTS), dtype=int)
for i in range(NUM_NURSES):
    for k in range(NUM_DAYS):
        # 各日に0-1のシフトを希望すると仮定
        requested_shift = np.random.randint(0, NUM_SHIFTS)
        requests[i, k, requested_shift] = 1

# コスト係数
# シフトごとのコスト (休みは0、他は1とする)
shift_costs = [0, 1, 1, 1]

# ペナルティの重み
lambda1 = 1  # 最大労働日数制約のペナルティ
lambda2 = 1  # 連続夜勤制限のペナルティ
lambda3 = 1  # 希望シフト違反のペナルティ

def solve_nurse_scheduling():
    # ソルバーの初期化（SCIPを使用）
    solver = pywraplp.Solver.CreateSolver("SCIP")
    
    if not solver:
        print("SCIP solver is not available.")
        return None
    
    # 変数の定義
    # x[i, k, s] = 看護師iが日kにシフトsに割り当てられるかどうか
    x = {}
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            for s in range(NUM_SHIFTS):
                x[i, k, s] = solver.BoolVar(f'x[{i},{k},{s}]')
    
    # y_plus[i] = 看護師iの最大労働日数を超える日数
    y_plus = {}
    for i in range(NUM_NURSES):
        y_plus[i] = solver.NumVar(0, solver.infinity(), f'y_plus[{i}]')
    
    # z[i, k] = 看護師iの日kからの連続夜勤違反
    z = {}
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS - 3):  # 連続4日間をチェックするため
            z[i, k] = solver.NumVar(0, solver.infinity(), f'z[{i},{k}]')
    
    # w[i, k, s] = 看護師iの日kのシフトsに対する希望違反
    w = {}
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            for s in range(NUM_SHIFTS):
                w[i, k, s] = solver.NumVar(0, 1, f'w[{i},{k},{s}]')
    
    # 制約の追加
    
    # 需要充足制約: 各日、各シフト（休み以外）の看護師数が需要を満たす
    for k in range(NUM_DAYS):
        for s in range(1, NUM_SHIFTS):  # 休み以外のシフト
            solver.Add(
                sum(x[i, k, s] for i in range(NUM_NURSES)) >= demands[k, s],
                f'demand_constraint[{k},{s}]'
            )
    
    # 日次シフト単一制約: 各看護師は各日に最大1つのシフトを担当
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            solver.Add(
                sum(x[i, k, s] for s in range(NUM_SHIFTS)) <= 1,
                f'single_shift_constraint[{i},{k}]'
            )
    
    # 連続勤務禁止制約: 夜勤の後は早番に入れない
    night_shift = 3  # 夜勤のインデックス
    early_shift = 1  # 早番のインデックス
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS - 1):  # 最終日の次の日はないので除外
            solver.Add(
                x[i, k, night_shift] + x[i, k+1, early_shift] <= 1,
                f'consecutive_shift_constraint[{i},{k}]'
            )
    
    # 最低休息時間制約: 夜勤後、10日中に最大10勤務まで
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS - 11):  # 11日先までをチェック
            constraint_expr = sum(
                x[i, k+t, s] for t in range(12) for s in range(1, NUM_SHIFTS)
            )
            solver.Add(
                constraint_expr <= 10,
                f'min_rest_constraint[{i},{k}]'
            )
    
    # ソフト制約
    
    # 最大労働日数制約
    for i in range(NUM_NURSES):
        solver.Add(
            sum(x[i, k, s] for k in range(NUM_DAYS) for s in range(1, NUM_SHIFTS)) - y_plus[i] <= max_shifts[i],
            f'max_shifts_constraint[{i}]'
        )
    
    # 希望シフト違反
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            for s in range(NUM_SHIFTS):
                if requests[i, k, s] == 1:  # シフトが希望されている場合
                    solver.Add(
                        w[i, k, s] >= requests[i, k, s] - x[i, k, s],
                        f'request_violation[{i},{k},{s}]'
                    )
    
    # 連続夜勤制限
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS - 3):  # 連続4日間をチェック
            solver.Add(
                z[i, k] >= sum(x[i, k+t, night_shift] for t in range(4)) - 3,
                f'consecutive_night_constraint[{i},{k}]'
            )
    
    # 目的関数の定義
    objective = solver.Objective()
    
    # 基本コスト
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            for s in range(NUM_SHIFTS):
                objective.SetCoefficient(x[i, k, s], shift_costs[s])
    
    # ペナルティ項
    # 最大労働日数超過ペナルティ
    for i in range(NUM_NURSES):
        objective.SetCoefficient(y_plus[i], lambda1)
    
    # 連続夜勤制限違反ペナルティ
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS - 3):
            objective.SetCoefficient(z[i, k], lambda2)
    
    # 希望シフト違反ペナルティ
    for i in range(NUM_NURSES):
        for k in range(NUM_DAYS):
            for s in range(NUM_SHIFTS):
                objective.SetCoefficient(w[i, k, s], lambda3)
    
    # 最小化問題として設定
    objective.SetMinimization()
    
    # 解を求める
    print("Solving the problem...")
    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()
    print(f"Solved in {end_time - start_time:.2f} seconds")
    
    # 結果の表示
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Solution found with objective value = {objective.Value()}")
        
        # 結果をデータフレームに格納
        schedule = np.zeros((NUM_NURSES, NUM_DAYS), dtype=int)
        for i in range(NUM_NURSES):
            for k in range(NUM_DAYS):
                for s in range(NUM_SHIFTS):
                    if x[i, k, s].solution_value() > 0.5:  # 四捨五入のための閾値
                        schedule[i, k] = s
        
        # ペナルティの表示
        total_shift_cost = sum(
            shift_costs[s] * x[i, k, s].solution_value()
            for i in range(NUM_NURSES)
            for k in range(NUM_DAYS)
            for s in range(NUM_SHIFTS)
        )
        
        total_y_plus = sum(y_plus[i].solution_value() for i in range(NUM_NURSES))
        
        total_z = sum(
            z[i, k].solution_value()
            for i in range(NUM_NURSES)
            for k in range(NUM_DAYS - 3)
        )
        
        total_w = sum(
            w[i, k, s].solution_value()
            for i in range(NUM_NURSES)
            for k in range(NUM_DAYS)
            for s in range(NUM_SHIFTS)
        )
        
        print(f"基本コスト: {total_shift_cost}")
        print(f"最大労働日数超過ペナルティ: {lambda1 * total_y_plus}")
        print(f"連続夜勤制限違反ペナルティ: {lambda2 * total_z}")
        print(f"希望シフト違反ペナルティ: {lambda3 * total_w}")
        
        return schedule
    else:
        print("No solution found.")
        return None

def display_schedule(schedule):
    """スケジュールを見やすく表示する"""
    # 整数値のスケジュールをコピー
    schedule_int = np.copy(schedule)
    
    # 新しいデータフレームを作成（文字列型）
    df = pd.DataFrame(
        [[SHIFT_NAMES[schedule_int[i, k]] for k in range(NUM_DAYS)] for i in range(NUM_NURSES)]
    )
    
    df.index = [f"看護師{i+1}" for i in range(NUM_NURSES)]
    df.columns = [f"日{k+1}" for k in range(NUM_DAYS)]
    
    return df

def main():
    schedule = solve_nurse_scheduling()
    if schedule is not None:
        df_schedule = display_schedule(schedule)
        print("\n最終スケジュール:")
        print(df_schedule)
        
        # 各看護師の統計情報
        nurse_stats = []
        for i in range(NUM_NURSES):
            shifts_worked = sum(1 for k in range(NUM_DAYS) if schedule[i, k] != 0)
            early_shifts = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 1)
            late_shifts = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 2)
            night_shifts = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 3)
            
            nurse_stats.append({
                "看護師": f"看護師{i+1}",
                "総勤務日数": shifts_worked,
                "早番": early_shifts,
                "遅番": late_shifts,
                "夜勤": night_shifts
            })
        
        df_stats = pd.DataFrame(nurse_stats)
        print("\n看護師別統計:")
        print(df_stats)
        
        # 各日の統計情報
        day_stats = []
        for k in range(NUM_DAYS):
            nurses_working = sum(1 for i in range(NUM_NURSES) if schedule[i, k] != 0)
            early_nurses = sum(1 for i in range(NUM_NURSES) if schedule[i, k] == 1)
            late_nurses = sum(1 for i in range(NUM_NURSES) if schedule[i, k] == 2)
            night_nurses = sum(1 for i in range(NUM_NURSES) if schedule[i, k] == 3)
            
            day_stats.append({
                "日付": f"日{k+1}",
                "勤務看護師数": nurses_working,
                "早番": early_nurses,
                "遅番": late_nurses,
                "夜勤": night_nurses
            })
        
        df_day_stats = pd.DataFrame(day_stats)
        print("\n日別統計:")
        print(df_day_stats)

if __name__ == "__main__":
    main()