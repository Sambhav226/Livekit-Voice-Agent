import random
from datetime import datetime, timedelta
import time

# --------------------------
# Dynamic Database Generator
# --------------------------
class DynamicUserDatabase:
    def __init__(self):
        self.database = self._generate_complete_database()
    
    def _generate_complete_database(self):
        """Generate a comprehensive database with random user data"""
        
        # Generate random user profiles
        first_names = ["Rajesh", "Priya", "Arjun", "Sneha", "Vikram", "Anita", "Rohit", "Kavya", "Suresh", "Meera"]
        last_names = ["Kumar", "Sharma", "Patel", "Singh", "Gupta", "Reddy", "Nair", "Joshi", "Yadav", "Mehta"]
        departments = ["Operations", "Manufacturing", "Quality Control", "Logistics", "Maintenance", "Security", "Admin"]
        shift_types = ["morning", "evening", "night", "double"]
        
        # Generate 5 random employees
        employees = []
        for i in range(5):
            emp_id = f"EMP{str(i+1).zfill(3)}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            full_name = f"{first_name} {last_name}"
            
            # Generate hire date (1-3 years ago)
            hire_date = datetime.now() - timedelta(days=random.randint(365, 1095))
            
            # Generate wage (1000-2500 per shift)
            daily_wage = random.randint(1000, 2500)
            
            # Generate shifts for current month (June 2025)
            shifts = []
            total_days_worked = random.randint(15, 22)  # 15-22 working days
            
            for day in range(1, total_days_worked + 1):
                if random.random() > 0.1:  # 90% chance of working each selected day
                    shift_date = f"2025-06-{str(day).zfill(2)}"
                    shift_type = random.choice(shift_types)
                    hours = 8 if shift_type != "double" else 16
                    
                    shifts.append({
                        "date": shift_date,
                        "shift_type": shift_type,
                        "hours": hours,
                        "status": "completed",
                        "overtime_hours": random.randint(0, 2) if random.random() > 0.7 else 0
                    })
            
            # Generate salary claims
            claims = []
            total_shifts = len(shifts)
            
            # First claim (approved)
            first_claim_shifts = random.randint(5, min(10, total_shifts))
            claims.append({
                "claim_id": f"CLM{emp_id}{random.randint(100, 999)}",
                "claim_date": "2025-06-10",
                "shifts_claimed": first_claim_shifts,
                "amount_claimed": first_claim_shifts * daily_wage,
                "status": "approved",
                "processed_date": "2025-06-12"
            })
            
            # Second claim (pending or approved)
            remaining_shifts = total_shifts - first_claim_shifts
            if remaining_shifts > 0:
                status = random.choice(["pending", "approved"])
                claims.append({
                    "claim_id": f"CLM{emp_id}{random.randint(100, 999)}",
                    "claim_date": "2025-06-18",
                    "shifts_claimed": remaining_shifts,
                    "amount_claimed": remaining_shifts * daily_wage,
                    "status": status,
                    "processed_date": "2025-06-20" if status == "approved" else None
                })
            
            # Generate performance metrics
            performance = {
                "attendance_rate": round(random.uniform(85, 98), 2),
                "punctuality_score": round(random.uniform(80, 100), 2),
                "quality_rating": round(random.uniform(3.5, 5.0), 1),
                "safety_incidents": random.randint(0, 2),
                "overtime_hours_month": sum([shift.get("overtime_hours", 0) for shift in shifts])
            }
            
            # Generate bank details
            bank_names = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Bank"]
            bank_details = {
                "account_number": f"****{random.randint(1000, 9999)}",
                "bank_name": random.choice(bank_names),
                "ifsc": f"{random.choice(['HDFC', 'ICIC', 'SBIN', 'UTIB', 'KKBK'])}000{random.randint(1000, 9999)}",
                "account_type": "Savings"
            }
            
            # Generate contact info
            contact = {
                "phone": f"+91-{random.randint(7000000000, 9999999999)}",
                "email": f"{first_name.lower()}.{last_name.lower()}@company.com",
                "address": f"{random.randint(1, 999)} {random.choice(['MG Road', 'Brigade Road', 'Whitefield', 'Koramangala', 'Indiranagar'])}, Bangalore"
            }
            
            # Generate leave records
            leaves = []
            for _ in range(random.randint(1, 4)):
                leave_types = ["sick", "casual", "emergency", "personal"]
                leave_date = datetime.now() - timedelta(days=random.randint(1, 180))
                leaves.append({
                    "leave_date": leave_date.strftime("%Y-%m-%d"),
                    "leave_type": random.choice(leave_types),
                    "status": random.choice(["approved", "pending"]),
                    "reason": "Personal work" if random.random() > 0.5 else "Medical"
                })
            
            employee = {
                "employee_id": emp_id,
                "name": full_name,
                "first_name": first_name,
                "last_name": last_name,
                "department": random.choice(departments),
                "designation": random.choice(["Operator", "Technician", "Supervisor", "Assistant", "Specialist"]),
                "hire_date": hire_date.strftime("%Y-%m-%d"),
                "daily_wage": daily_wage,
                "current_month": "June 2025",
                "shifts_worked": shifts,
                "salary_claims": claims,
                "performance_metrics": performance,
                "bank_details": bank_details,
                "contact_info": contact,
                "leave_records": leaves,
                "total_earnings_ytd": random.randint(80000, 200000),
                "employee_status": "active",
                "contract_type": random.choice(["permanent", "contract", "temporary"])
            }
            
            employees.append(employee)
        
        # Generate company-wide statistics
        company_stats = {
            "total_employees": len(employees),
            "total_shifts_month": sum([len(emp["shifts_worked"]) for emp in employees]),
            "total_payroll_month": sum([
                sum([claim["amount_claimed"] for claim in emp["salary_claims"]]) 
                for emp in employees
            ]),
            "pending_claims_count": sum([
                len([claim for claim in emp["salary_claims"] if claim["status"] == "pending"]) 
                for emp in employees
            ]),
            "average_daily_wage": sum([emp["daily_wage"] for emp in employees]) / len(employees),
            "departments": list(set([emp["department"] for emp in employees]))
        }
        
        return {
            "employees": employees,
            "company_stats": company_stats,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database_version": "1.0"
        }
    
    def get_employee_by_id(self, emp_id: str):
        """Get employee data by ID"""
        for emp in self.database["employees"]:
            if emp["employee_id"] == emp_id:
                return emp
        return None
    
    def get_employee_by_name(self, name: str):
        """Get employee data by name (partial match)"""
        name_lower = name.lower()
        for emp in self.database["employees"]:
            if name_lower in emp["name"].lower():
                return emp
        return None
    
    def get_all_employees(self):
        """Get all employee records"""
        return self.database["employees"]
    
    def get_company_stats(self):
        """Get company-wide statistics"""
        return self.database["company_stats"]
    
    def search_employees(self, criteria: dict):
        """Search employees based on criteria"""
        results = []
        for emp in self.database["employees"]:
            match = True
            for key, value in criteria.items():
                if key in emp and str(emp[key]).lower() != str(value).lower():
                    match = False
                    break
            if match:
                results.append(emp)
        return results