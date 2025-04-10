from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
import time
import calendar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import japanize_matplotlib

# 問題の定数
NUM_ENGINEERS = 10    # エンジニアの数
NUM_DAYS = 180        # スケジューリング期間（6ヶ月≒180日）
NUM_SHIFTS = 2        # シフトの種類: 0:オンコールなし, 1:オンコール当番

# シフトの名称
SHIFT_NAMES = ["なし", "当番"]

# 日付情報の作成
start_date = datetime(2025, 5, 1)  # 開始日を2025年5月1日とする
dates = [start_date + timedelta(days=i) for i in range(NUM_DAYS)]
date_strs = [d.strftime('%Y-%m-%d') for d in dates]
weekday_names = ['月', '火', '水', '木', '金', '土', '日']
date_display = [f"{d.strftime('%m/%d')}({weekday_names[d.weekday()]})" for d in dates]

# 平日と週末を区別（0:平日, 1:週末）
is_weekend = np.zeros(NUM_DAYS, dtype=int)
for i in range(NUM_DAYS):
    # 週末（土日）を設定
    if dates[i].weekday() >= 5:  # 5が土曜、6が日曜
        is_weekend[i] = 1

# 祝日設定（日本の祝日の例、実際には祝日リストを用意する必要あり）
is_holiday = np.zeros(NUM_DAYS, dtype=int)
# 例: 祝日を適宜追加
holiday_dates = [
    "2025-05-03", "2025-05-04", "2025-05-05", "2025-07-21", 
    "2025-08-11", "2025-09-15", "2025-09-23", "2025-10-13"
]
for i, date_str in enumerate(date_strs):
    if date_str in holiday_dates:
        is_holiday[i] = 1

# 各エンジニアの最大オンコール日数（1ヶ月あたり約5日を6ヶ月分）
max_shifts_per_month = 5
max_shifts = np.full(NUM_ENGINEERS, max_shifts_per_month * 6)

# 各エンジニアの希望シフト (1: 希望する, 0: 希望しない)
np.random.seed(42)
requests = np.zeros((NUM_ENGINEERS, NUM_DAYS, NUM_SHIFTS), dtype=int)
for i in range(NUM_ENGINEERS):
    # 各エンジニアが希望休暇を設定（ランダムに20日程度）
    vacation_days = np.random.choice(range(NUM_DAYS), size=20, replace=False)
    for k in vacation_days:
        requests[i, k, 0] = 1  # 休暇希望日はオンコールなしを希望

# コスト係数
# シフトごとのコスト (オンコールなしは0、オンコール当番は1)
shift_costs = [0, 1]

# ペナルティの重み
lambda1 = 2   # 最大オンコール日数制約のペナルティ
lambda2 = 5   # 連続オンコール制限のペナルティ
lambda3 = 3   # 希望シフト違反のペナルティ
lambda4 = 2   # 週末・祝日担当バランスのペナルティ
lambda5 = 3   # 月ごとのバランスのペナルティ
lambda6 = 4   # バックアップ制約のペナルティ

def solve_oncall_scheduling():
    # ソルバーの初期化（SCIPを使用）
    solver = pywraplp.Solver.CreateSolver("SCIP")
    
    if not solver:
        print("SCIP solver is not available.")
        return None
    
    # 変数の定義
    # x[i, k] = エンジニアiが日kにオンコール当番に割り当てられるかどうか
    x = {}
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            x[i, k] = solver.BoolVar(f'x[{i},{k}]')
    
    # バックアップ担当変数（各当番に対してバックアップを1人割り当て）
    backup = {}
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            backup[i, k] = solver.BoolVar(f'backup[{i},{k}]')
    
    # y_plus[i] = エンジニアiの最大オンコール日数を超える日数
    y_plus = {}
    for i in range(NUM_ENGINEERS):
        y_plus[i] = solver.NumVar(0, solver.infinity(), f'y_plus[{i}]')
    
    # z[i, k] = エンジニアiの日kからの連続オンコール違反
    z = {}
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 2):  # 連続3日間をチェックするため
            z[i, k] = solver.NumVar(0, solver.infinity(), f'z[{i},{k}]')
    
    # w[i, k] = エンジニアiの日kに対する希望違反
    w = {}
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            w[i, k] = solver.NumVar(0, 1, f'w[{i},{k}]')
    
    # u[i] = エンジニアiの週末・祝日オンコール担当回数の不均衡
    u = {}
    for i in range(NUM_ENGINEERS):
        u[i] = solver.NumVar(0, solver.infinity(), f'u[{i}]')
    
    # 月ごとのバランス変数
    # v[i, m] = エンジニアiの月mにおけるオンコール担当回数の不均衡
    v = {}
    num_months = 6
    for i in range(NUM_ENGINEERS):
        for m in range(num_months):
            v[i, m] = solver.NumVar(0, solver.infinity(), f'v[{i},{m}]')
    
    # 制約の追加
    
    # 需要充足制約: 各日に必ず1人のオンコール当番が必要
    for k in range(NUM_DAYS):
        solver.Add(
            sum(x[i, k] for i in range(NUM_ENGINEERS)) == 1,
            f'demand_constraint[{k}]'
        )
    
    # バックアップ担当制約: 各日に必ず1人のバックアップが必要
    for k in range(NUM_DAYS):
        solver.Add(
            sum(backup[i, k] for i in range(NUM_ENGINEERS)) == 1,
            f'backup_constraint[{k}]'
        )
    
    # 同じ日に当番とバックアップを兼任できない
    for k in range(NUM_DAYS):
        for i in range(NUM_ENGINEERS):
            solver.Add(
                x[i, k] + backup[i, k] <= 1,
                f'no_double_duty[{i},{k}]'
            )
    
    # 当番とバックアップの連続性禁止（前日当番だったら翌日バックアップにしない）
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 1):
            solver.Add(
                x[i, k] + backup[i, k+1] <= 1,
                f'duty_backup_consecutive[{i},{k}]'
            )
    
    # ソフト制約
    
    # 最大オンコール日数制約（当番とバックアップを合わせて）
    for i in range(NUM_ENGINEERS):
        solver.Add(
            sum(x[i, k] + backup[i, k] for k in range(NUM_DAYS)) - y_plus[i] <= max_shifts[i],
            f'max_shifts_constraint[{i}]'
        )
    
    # 希望シフト違反（休暇希望日にオンコール当番やバックアップに入れない）
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            if requests[i, k, 0] == 1:  # 休暇希望日
                solver.Add(
                    w[i, k] >= requests[i, k, 0] - (1 - x[i, k] - backup[i, k]),
                    f'request_violation[{i},{k}]'
                )
    
    # 連続オンコール制限（当番とバックアップ合わせて）
    # 3日以上連続でオンコール当番またはバックアップにしない
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 2):  # 連続3日間をチェック
            solver.Add(
                z[i, k] >= (x[i, k] + backup[i, k] + x[i, k+1] + backup[i, k+1] + x[i, k+2] + backup[i, k+2]) - 2,
                f'consecutive_oncall_constraint[{i},{k}]'
            )
    
    # 週末・祝日オンコールバランス制約
    total_weekend_holidays = sum(is_weekend) + sum(is_holiday)
    avg_weekend_holiday_shifts = total_weekend_holidays / NUM_ENGINEERS  # 1人あたりの平均週末・祝日オンコール回数
    
    for i in range(NUM_ENGINEERS):
        weekend_holiday_shifts = sum((x[i, k] + 0.5 * backup[i, k]) * (is_weekend[k] or is_holiday[k]) for k in range(NUM_DAYS))
        solver.Add(
            u[i] >= weekend_holiday_shifts - avg_weekend_holiday_shifts,
            f'weekend_holiday_balance_plus[{i}]'
        )
        solver.Add(
            u[i] >= avg_weekend_holiday_shifts - weekend_holiday_shifts,
            f'weekend_holiday_balance_minus[{i}]'
        )
    
    # 月ごとのバランス制約
    # 月ごとの日数を計算
    month_days = {}
    for m in range(num_months):
        month_start = 0
        for prev_m in range(m):
            month = (start_date.month + prev_m - 1) % 12 + 1
            year = start_date.year + (start_date.month + prev_m - 1) // 12
            _, days_in_month = calendar.monthrange(year, month)
            month_start += days_in_month
        
        month = (start_date.month + m - 1) % 12 + 1
        year = start_date.year + (start_date.month + m - 1) // 12
        _, days_in_month = calendar.monthrange(year, month)
        month_end = min(month_start + days_in_month, NUM_DAYS)
        
        month_days[m] = list(range(month_start, month_end))
    
    # 各月、各エンジニアあたりの平均オンコール回数
    for m in range(num_months):
        if month_days[m]:  # 月に日があることを確認
            avg_monthly_shifts = len(month_days[m]) / NUM_ENGINEERS
            
            for i in range(NUM_ENGINEERS):
                monthly_shifts = sum(x[i, k] + 0.5 * backup[i, k] for k in month_days[m])
                
                solver.Add(
                    v[i, m] >= monthly_shifts - avg_monthly_shifts,
                    f'monthly_balance_plus[{i},{m}]'
                )
                solver.Add(
                    v[i, m] >= avg_monthly_shifts - monthly_shifts,
                    f'monthly_balance_minus[{i},{m}]'
                )
    
    # 休息時間制約: オンコール後、少なくとも1日の休息
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 1):
            solver.Add(
                x[i, k] + x[i, k+1] <= 1,
                f'rest_time_constraint[{i},{k}]'
            )
    
    # 目的関数の定義
    objective = solver.Objective()
    
    # 基本コスト
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            objective.SetCoefficient(x[i, k], shift_costs[1])  # オンコール当番コスト
            objective.SetCoefficient(backup[i, k], shift_costs[1] * 0.5)  # バックアップは当番の半分のコスト
    
    # ペナルティ項
    # 最大オンコール日数超過ペナルティ
    for i in range(NUM_ENGINEERS):
        objective.SetCoefficient(y_plus[i], lambda1)
    
    # 連続オンコール制限違反ペナルティ
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 2):
            objective.SetCoefficient(z[i, k], lambda2)
    
    # 希望シフト違反ペナルティ
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS):
            objective.SetCoefficient(w[i, k], lambda3)
    
    # 週末・祝日オンコールバランス違反ペナルティ
    for i in range(NUM_ENGINEERS):
        objective.SetCoefficient(u[i], lambda4)
    
    # 月ごとのバランス違反ペナルティ
    for i in range(NUM_ENGINEERS):
        for m in range(num_months):
            objective.SetCoefficient(v[i, m], lambda5)
    
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
        
        # 結果を配列に格納
        schedule = np.zeros((NUM_ENGINEERS, NUM_DAYS), dtype=int)
        backup_schedule = np.zeros((NUM_ENGINEERS, NUM_DAYS), dtype=int)
        
        for i in range(NUM_ENGINEERS):
            for k in range(NUM_DAYS):
                if x[i, k].solution_value() > 0.5:
                    schedule[i, k] = 1
                if backup[i, k].solution_value() > 0.5:
                    backup_schedule[i, k] = 1
        
        # ペナルティの表示
        total_shift_cost = sum(
            shift_costs[1] * x[i, k].solution_value() + 0.5 * shift_costs[1] * backup[i, k].solution_value()
            for i in range(NUM_ENGINEERS)
            for k in range(NUM_DAYS)
        )
        
        total_y_plus = sum(y_plus[i].solution_value() for i in range(NUM_ENGINEERS))
        
        total_z = sum(
            z[i, k].solution_value()
            for i in range(NUM_ENGINEERS)
            for k in range(NUM_DAYS - 2)
        )
        
        total_w = sum(
            w[i, k].solution_value()
            for i in range(NUM_ENGINEERS)
            for k in range(NUM_DAYS)
        )
        
        total_u = sum(u[i].solution_value() for i in range(NUM_ENGINEERS))
        
        total_v = sum(
            v[i, m].solution_value()
            for i in range(NUM_ENGINEERS)
            for m in range(num_months)
        )
        
        print(f"基本コスト: {total_shift_cost}")
        print(f"最大オンコール日数超過ペナルティ: {lambda1 * total_y_plus}")
        print(f"連続オンコール制限違反ペナルティ: {lambda2 * total_z}")
        print(f"希望シフト違反ペナルティ: {lambda3 * total_w}")
        print(f"週末・祝日担当バランス違反ペナルティ: {lambda4 * total_u}")
        print(f"月ごとのバランス違反ペナルティ: {lambda5 * total_v}")
        
        return schedule, backup_schedule
    else:
        print("No solution found.")
        return None, None

def display_schedule(schedule, backup_schedule):
    """スケジュールを見やすく表示する"""
    # 月ごとにデータフレームを分割して表示
    current_month = start_date.month
    current_year = start_date.year
    month_dfs = []
    
    month_start_idx = 0
    for month_offset in range(6):  # 6ヶ月分
        month = (current_month + month_offset - 1) % 12 + 1
        year = current_year + (current_month + month_offset - 1) // 12
        _, days_in_month = calendar.monthrange(year, month)
        
        month_end_idx = min(month_start_idx + days_in_month, NUM_DAYS)
        month_dates = date_display[month_start_idx:month_end_idx]
        
        # 当番スケジュール
        duty_data = []
        for i in range(NUM_ENGINEERS):
            row = []
            for k in range(month_start_idx, month_end_idx):
                if schedule[i, k] == 1:
                    row.append("当番")
                elif backup_schedule[i, k] == 1:
                    row.append("バックアップ")
                else:
                    row.append("")
            duty_data.append(row)
        
        df = pd.DataFrame(duty_data, columns=month_dates)
        df.index = [f"エンジニア{i+1}" for i in range(NUM_ENGINEERS)]
        
        month_name = f"{year}年{month}月"
        month_dfs.append((month_name, df))
        
        month_start_idx = month_end_idx
    
    return month_dfs

def get_engineer_stats(schedule, backup_schedule):
    """各エンジニアの統計情報を計算する"""
    engineer_stats = []
    
    for i in range(NUM_ENGINEERS):
        duty_days = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 1)
        backup_days = sum(1 for k in range(NUM_DAYS) if backup_schedule[i, k] == 1)
        weekend_duty = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 1 and is_weekend[k] == 1)
        holiday_duty = sum(1 for k in range(NUM_DAYS) if schedule[i, k] == 1 and is_holiday[k] == 1)
        
        # 月ごとの当番回数
        monthly_duties = {}
        monthly_backups = {}
        
        current_month = start_date.month
        current_year = start_date.year
        month_start_idx = 0
        
        for month_offset in range(6):  # 6ヶ月分
            month = (current_month + month_offset - 1) % 12 + 1
            year = current_year + (current_month + month_offset - 1) // 12
            _, days_in_month = calendar.monthrange(year, month)
            
            month_end_idx = min(month_start_idx + days_in_month, NUM_DAYS)
            month_name = f"{year}年{month}月"
            
            monthly_duties[month_name] = sum(1 for k in range(month_start_idx, month_end_idx) if schedule[i, k] == 1)
            monthly_backups[month_name] = sum(1 for k in range(month_start_idx, month_end_idx) if backup_schedule[i, k] == 1)
            
            month_start_idx = month_end_idx
        
        engineer_stats.append({
            "エンジニア": f"エンジニア{i+1}",
            "総当番日数": duty_days,
            "総バックアップ日数": backup_days,
            "週末当番": weekend_duty,
            "祝日当番": holiday_duty,
            "月別当番": monthly_duties,
            "月別バックアップ": monthly_backups
        })
    
    return pd.DataFrame(engineer_stats)

def analyze_schedule_results(schedule, backup_schedule, requests):
    """スケジュール結果の詳細な分析を行う"""
    print("\n===== スケジュール分析結果 =====")
    
    # 負荷バランス分析
    duty_counts = [sum(schedule[i]) for i in range(NUM_ENGINEERS)]
    backup_counts = [sum(backup_schedule[i]) for i in range(NUM_ENGINEERS)]
    total_load = [duty_counts[i] + 0.5*backup_counts[i] for i in range(NUM_ENGINEERS)]
    
    print(f"\n1. 負荷バランス分析:")
    print(f"   当番回数: 最小={min(duty_counts)}, 最大={max(duty_counts)}, 平均={sum(duty_counts)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(duty_counts):.2f}")
    print(f"   バックアップ回数: 最小={min(backup_counts)}, 最大={max(backup_counts)}, 平均={sum(backup_counts)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(backup_counts):.2f}")
    print(f"   総負荷(当番+0.5*バックアップ): 最小={min(total_load):.1f}, 最大={max(total_load):.1f}, 平均={sum(total_load)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(total_load):.2f}")
    
    # 連続当番/バックアップ分析
    consecutive_duties = []
    for i in range(NUM_ENGINEERS):
        max_consecutive = 0
        current_consecutive = 0
        for k in range(NUM_DAYS):
            if schedule[i, k] == 1 or backup_schedule[i, k] == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        consecutive_duties.append(max_consecutive)
    
    print(f"\n2. 連続当番分析:")
    print(f"   最大連続当番日数: 最小={min(consecutive_duties)}, 最大={max(consecutive_duties)}, 平均={sum(consecutive_duties)/NUM_ENGINEERS:.2f}")
    
    # 週末・祝日負荷分析
    weekend_holiday_counts = []
    for i in range(NUM_ENGINEERS):
        count = sum(schedule[i, k] for k in range(NUM_DAYS) if is_weekend[k] or is_holiday[k])
        weekend_holiday_counts.append(count)
    
    total_weekend_holidays = sum(is_weekend) + sum(is_holiday)
    print(f"\n3. 週末・祝日負荷分析:")
    print(f"   週末・祝日の総数: {total_weekend_holidays}")
    print(f"   週末・祝日担当回数: 最小={min(weekend_holiday_counts)}, 最大={max(weekend_holiday_counts)}, 平均={sum(weekend_holiday_counts)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(weekend_holiday_counts):.2f}")
    
    # 希望シフト遵守率分析
    request_violations = []
    for i in range(NUM_ENGINEERS):
        violations = 0
        for k in range(NUM_DAYS):
            if requests[i, k, 0] == 1 and (schedule[i, k] == 1 or backup_schedule[i, k] == 1):
                violations += 1
        request_violations.append(violations)
    
    total_requests = sum(sum(requests[i, :, 0]) for i in range(NUM_ENGINEERS))
    total_violations = sum(request_violations)
    if total_requests > 0:
        compliance_rate = 100 * (1 - total_violations / total_requests)
    else:
        compliance_rate = 100
    
    print(f"\n4. 希望シフト遵守率分析:")
    print(f"   総希望休暇日数: {total_requests}")
    print(f"   違反回数: {total_violations}")
    print(f"   全体遵守率: {compliance_rate:.2f}%")
    print(f"   エンジニア別違反回数: 最小={min(request_violations)}, 最大={max(request_violations)}, 平均={sum(request_violations)/NUM_ENGINEERS:.2f}")
    
    # 月ごとの負荷バランス分析
    print(f"\n5. 月ごとの負荷バランス分析:")
    current_month = start_date.month
    current_year = start_date.year
    month_start_idx = 0
    
    for month_offset in range(6):  # 6ヶ月分
        month = (current_month + month_offset - 1) % 12 + 1
        year = current_year + (current_month + month_offset - 1) // 12
        _, days_in_month = calendar.monthrange(year, month)
        
        month_end_idx = min(month_start_idx + days_in_month, NUM_DAYS)
        month_name = f"{year}年{month}月"
        
        month_duties = []
        month_backups = []
        month_loads = []
        
        for i in range(NUM_ENGINEERS):
            duty_count = sum(schedule[i, k] for k in range(month_start_idx, month_end_idx))
            backup_count = sum(backup_schedule[i, k] for k in range(month_start_idx, month_end_idx))
            total_load = duty_count + 0.5 * backup_count
            
            month_duties.append(duty_count)
            month_backups.append(backup_count)
            month_loads.append(total_load)
        
        print(f"   {month_name}:")
        print(f"     当番回数: 最小={min(month_duties)}, 最大={max(month_duties)}, 平均={sum(month_duties)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(month_duties):.2f}")
        print(f"     バックアップ回数: 最小={min(month_backups)}, 最大={max(month_backups)}, 平均={sum(month_backups)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(month_backups):.2f}")
        print(f"     総負荷: 最小={min(month_loads):.1f}, 最大={max(month_loads):.1f}, 平均={sum(month_loads)/NUM_ENGINEERS:.2f}, 標準偏差={np.std(month_loads):.2f}")
        
        month_start_idx = month_end_idx
    
    # 最適化係数に関する分析
    print(f"\n6. 最適化効率分析:")
    
    # 最大オンコール日数制約のチェック
    max_shifts_violations = [max(0, sum(schedule[i]) + sum(backup_schedule[i]) - max_shifts[i]) for i in range(NUM_ENGINEERS)]
    print(f"   最大オンコール日数制約違反: {sum(max_shifts_violations)}")
    
    # 連続オンコール制約のチェック
    consecutive_violations = 0
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 2):
            if (schedule[i, k] + backup_schedule[i, k] + 
                schedule[i, k+1] + backup_schedule[i, k+1] + 
                schedule[i, k+2] + backup_schedule[i, k+2]) > 2:
                consecutive_violations += 1
    
    print(f"   連続オンコール制約違反: {consecutive_violations}")
    
    # 当番-バックアップ連続性のチェック
    duty_backup_violations = 0
    for i in range(NUM_ENGINEERS):
        for k in range(NUM_DAYS - 1):
            if schedule[i, k] == 1 and backup_schedule[i, k+1] == 1:
                duty_backup_violations += 1
    
    print(f"   当番-バックアップ連続性違反: {duty_backup_violations}")
    
    # 視覚化のためのデータを返す（必要に応じて）
    return {
        "duty_counts": duty_counts,
        "backup_counts": backup_counts,
        "total_load": total_load,
        "weekend_holiday_counts": weekend_holiday_counts,
        "request_violations": request_violations
    }

# グラフ可視化関数（オプション）
def visualize_schedule_results(analysis_data):
    # エンジニア別負荷グラフ
    plt.figure(figsize=(12, 6))
    engineers = [f"E{i+1}" for i in range(NUM_ENGINEERS)]
    
    # duty_countsとbackup_countsからtotal_loadを計算
    duty_counts = analysis_data["duty_counts"]
    backup_counts = analysis_data["backup_counts"]
    total_load = [duty_counts[i] + 0.5 * backup_counts[i] for i in range(len(duty_counts))]
    
    plt.bar(engineers, duty_counts, label='当番')
    plt.bar(engineers, [0.5 * bc for bc in backup_counts], 
            bottom=duty_counts, label='バックアップ(0.5倍)')
    
    # 平均負荷を計算
    average_load = sum(total_load) / len(total_load)
    plt.axhline(y=average_load, color='r', linestyle='-', label='平均負荷')
    
    plt.xlabel('エンジニア')
    plt.ylabel('負荷')
    plt.title('エンジニア別オンコール負荷分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('engineer_load_distribution.png')
    plt.close()
    
    # 週末・祝日負荷分布
    plt.figure(figsize=(12, 6))
    plt.bar(engineers, analysis_data["weekend_holiday_counts"])
    plt.axhline(y=sum(analysis_data["weekend_holiday_counts"])/NUM_ENGINEERS, color='r', linestyle='-', label='平均')
    
    plt.xlabel('エンジニア')
    plt.ylabel('回数')
    plt.title('エンジニア別週末・祝日当番回数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('weekend_holiday_distribution.png')
    plt.close()
    
    # 希望違反回数
    plt.figure(figsize=(12, 6))
    plt.bar(engineers, analysis_data["request_violations"])
    
    plt.xlabel('エンジニア')
    plt.ylabel('違反回数')
    plt.title('エンジニア別希望シフト違反回数')
    plt.grid(True, alpha=0.3)
    plt.savefig('request_violations.png')
    plt.close()
    
    print("分析グラフを保存しました。")

def main():
    schedule, backup_schedule = solve_oncall_scheduling()
    
    if schedule is not None and backup_schedule is not None:
        month_dfs = display_schedule(schedule, backup_schedule)
        
        print("\n6ヶ月間のオンコールスケジュール:")
        for month_name, df in month_dfs:
            print(f"\n===== {month_name} =====")
            print(df)
        
        # 各エンジニアの統計情報
        df_stats = get_engineer_stats(schedule, backup_schedule)
        print("\nエンジニア別統計:")
        print(df_stats[["エンジニア", "総当番日数", "総バックアップ日数", "週末当番", "祝日当番"]])
        
        # 月ごとの統計情報を別途表示
        print("\n月別担当回数:")
        for month_offset in range(6):
            month = (start_date.month + month_offset - 1) % 12 + 1
            year = start_date.year + (start_date.month + month_offset - 1) // 12
            month_name = f"{year}年{month}月"
            
            month_stats = {
                "エンジニア": [f"エンジニア{i+1}" for i in range(NUM_ENGINEERS)],
                "当番回数": [df_stats["月別当番"].iloc[i][month_name] for i in range(NUM_ENGINEERS)],
                "バックアップ回数": [df_stats["月別バックアップ"].iloc[i][month_name] for i in range(NUM_ENGINEERS)]
            }
            
            month_df = pd.DataFrame(month_stats)
            print(f"\n{month_name}:")
            print(month_df)
                # 詳細な分析結果を表示
        analysis_data = analyze_schedule_results(schedule, backup_schedule, requests)
        
        # オプション: 分析結果をグラフとして視覚化
        visualize_schedule_results(analysis_data)

if __name__ == "__main__":
    main()