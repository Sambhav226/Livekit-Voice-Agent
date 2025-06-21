# mcp_server.py
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sop-mcp-server")

from VectorSearch import VectorSearch


from DynamicUserDatabase import DynamicUserDatabase

# Initialize services
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "your-index-name")
vector_service = VectorSearch(index_name=PINECONE_INDEX)
user_database = DynamicUserDatabase()

# Create MCP server
mcp = FastMCP("SOP Database MCP Server")

async def retrieve_sop_docs(query: str, namespace: str, doc_category: str, topn: int):
    """
    Retrieve SOP documents asynchronously
    """
    try:
        docs = await vector_service.retrieval(
            query=query,
            namespace=namespace,
            doc_category=doc_category,
            topn=topn
        )
        return docs
    except Exception as e:
        logger.error(f"Error retrieving SOP documents: {str(e)}")
        return []

def analyze_sop_requirements(instructions: str, query: str):
    """
    Analyze SOP instructions to determine database access needs
    """
    clean_instructions = instructions.replace('*', '').replace('#', '').replace('- ', '')
    clean_instructions = ' '.join(clean_instructions.split())
    
    database_access_indicators = [
        "check employee", "verify employee", "employee records", "payroll data",
        "salary information", "shift data", "performance review", "attendance",
        "wage details", "employee profile", "worker information", "staff data",
        "employment details", "compensation review", "work history", 
        "employee search", "find employee", "employee lookup", "database query",
        "check database", "access records", "retrieve data", "employee analysis"
    ]
    
    requires_database_access = any(
        indicator.lower() in clean_instructions.lower() 
        for indicator in database_access_indicators
    )
    
    employee_query_indicators = [
        "employee", "worker", "staff", "payroll", "salary", "wage", "shift",
        "earnings", "performance", "attendance", "profile", "record"
    ]
    
    user_requests_employee_data = any(
        indicator.lower() in query.lower() 
        for indicator in employee_query_indicators
    )
    
    final_decision = requires_database_access or user_requests_employee_data
    
    found_indicators = [
        indicator for indicator in database_access_indicators 
        if indicator.lower() in clean_instructions.lower()
    ]
    
    return {
        "requires_database_access": final_decision,
        "database_indicators_found": found_indicators,
        "user_query_requires_data": user_requests_employee_data,
        "sop_requires_data": requires_database_access
    }

def perform_database_analysis(query: str, sop_context: str, all_employees: List[Dict], company_stats: Dict):
    """
    Perform comprehensive database analysis
    """
    analysis_log = ["Database analysis initiated per SOP requirements"]
    findings = {}
    
    query_lower = query.lower()
    sop_lower = sop_context.lower() if sop_context else ""
    combined_context = f"{query_lower} {sop_lower}"
    
    # Employee identification analysis
    if any(word in combined_context for word in ['employee', 'worker', 'staff', 'name', 'who', 'profile']):
        analysis_log.append("→ Employee identification analysis")
        
        target_employee = None
        for emp in all_employees:
            if emp["name"].lower() in query_lower or emp["employee_id"].lower() in query_lower:
                target_employee = emp
                break
        
        if target_employee:
            findings["target_employee"] = {
                "employee_id": target_employee["employee_id"],
                "name": target_employee["name"],
                "department": target_employee["department"],
                "designation": target_employee["designation"],
                "hire_date": target_employee["hire_date"],
                "status": target_employee["employee_status"],
                "contract_type": target_employee["contract_type"]
            }
            analysis_log.append(f"  ✓ Target employee identified: {target_employee['name']}")
        else:
            findings["employee_overview"] = {
                "total_employees": len(all_employees),
                "employees_list": [{"id": emp["employee_id"], "name": emp["name"], "dept": emp["department"]} for emp in all_employees[:5]],
                "departments": list(set([emp["department"] for emp in all_employees]))
            }
            analysis_log.append("  ✓ Employee overview generated")
    
    # Payroll analysis
    if any(word in combined_context for word in ['salary', 'wage', 'pay', 'earning', 'money', 'payroll', 'compensation']):
        analysis_log.append("→ Comprehensive payroll analysis")
        
        total_payroll = 0
        wage_distribution = []
        
        for emp in all_employees:
            emp_earnings = sum([claim["amount_claimed"] for claim in emp["salary_claims"]])
            total_payroll += emp_earnings
            wage_distribution.append({
                "employee": emp["name"],
                "daily_wage": emp["daily_wage"],
                "monthly_earnings": emp_earnings,
                "ytd_earnings": emp["total_earnings_ytd"]
            })
        
        findings["payroll_analysis"] = {
            "total_monthly_payroll": total_payroll,
            "average_daily_wage": company_stats["average_daily_wage"],
            "wage_distribution": wage_distribution,
            "highest_earner": max(wage_distribution, key=lambda x: x["monthly_earnings"]) if wage_distribution else None,
            "payroll_summary": f"₹{total_payroll:,} total monthly payroll across {len(all_employees)} employees"
        }
        analysis_log.append(f"  ✓ Payroll calculated: ₹{total_payroll:,} total")
    
    # Shift and attendance analysis
    if any(word in combined_context for word in ['shift', 'work', 'attendance', 'hours', 'schedule', 'time']):
        analysis_log.append("→ Shift and attendance analysis")
        
        total_shifts = 0
        total_hours = 0
        shift_patterns = {}
        attendance_records = []
        
        for emp in all_employees:
            emp_shifts = len(emp["shifts_worked"])
            emp_hours = sum([shift["hours"] for shift in emp["shifts_worked"]])
            total_shifts += emp_shifts
            total_hours += emp_hours
            
            for shift in emp["shifts_worked"]:
                shift_type = shift["shift_type"]
                shift_patterns[shift_type] = shift_patterns.get(shift_type, 0) + 1
            
            attendance_records.append({
                "employee": emp["name"],
                "shifts_worked": emp_shifts,
                "total_hours": emp_hours,
                "attendance_rate": emp["performance_metrics"]["attendance_rate"],
                "overtime_hours": emp["performance_metrics"]["overtime_hours_month"]
            })
        
        findings["attendance_analysis"] = {
            "total_company_shifts": total_shifts,
            "total_company_hours": total_hours,
            "shift_pattern_distribution": shift_patterns,
            "employee_attendance": attendance_records,
            "average_shifts_per_employee": total_shifts / len(all_employees) if all_employees else 0,
            "best_attendance": max(attendance_records, key=lambda x: x["attendance_rate"]) if attendance_records else None
        }
        analysis_log.append(f"  ✓ Analyzed {total_shifts} shifts totaling {total_hours} hours")
    
    # Performance metrics analysis
    if any(word in combined_context for word in ['performance', 'rating', 'quality', 'punctuality', 'review']):
        analysis_log.append("→ Performance metrics analysis")
        
        performance_data = []
        for emp in all_employees:
            metrics = emp["performance_metrics"]
            performance_data.append({
                "employee": emp["name"],
                "attendance_rate": metrics["attendance_rate"],
                "punctuality_score": metrics["punctuality_score"],
                "quality_rating": metrics["quality_rating"],
                "safety_incidents": metrics["safety_incidents"],
                "overall_score": (metrics["attendance_rate"] + metrics["punctuality_score"] + (metrics["quality_rating"] * 20)) / 3
            })
        
        findings["performance_analysis"] = {
            "performance_data": performance_data,
            "top_performer": max(performance_data, key=lambda x: x["overall_score"]) if performance_data else None,
            "average_quality_rating": sum([p["quality_rating"] for p in performance_data]) / len(performance_data) if performance_data else 0,
            "total_safety_incidents": sum([p["safety_incidents"] for p in performance_data])
        }
        analysis_log.append("  ✓ Performance metrics analyzed for all employees")
    
    # Claims and payments analysis
    if any(word in combined_context for word in ['claim', 'pending', 'payment', 'approve', 'process', 'status']):
        analysis_log.append("→ Claims and payments analysis")
        
        all_claims = []
        pending_claims = []
        approved_claims = []
        
        for emp in all_employees:
            for claim in emp["salary_claims"]:
                claim_data = {
                    "employee": emp["name"],
                    "employee_id": emp["employee_id"],
                    "claim_id": claim["claim_id"],
                    "amount": claim["amount_claimed"],
                    "status": claim["status"],
                    "claim_date": claim["claim_date"]
                }
                all_claims.append(claim_data)
                
                if claim["status"] == "pending":
                    pending_claims.append(claim_data)
                else:
                    approved_claims.append(claim_data)
        
        findings["claims_analysis"] = {
            "total_claims": len(all_claims),
            "pending_claims": pending_claims,
            "approved_claims": approved_claims,
            "pending_amount": sum([c["amount"] for c in pending_claims]),
            "approved_amount": sum([c["amount"] for c in approved_claims]),
            "claims_requiring_action": len(pending_claims)
        }
        analysis_log.append(f"  ✓ Found {len(pending_claims)} pending claims worth ₹{sum([c['amount'] for c in pending_claims]):,}")
    
    # Department-wise analysis
    if any(word in combined_context for word in ['department', 'team', 'division', 'group']):
        analysis_log.append("→ Department-wise analysis")
        
        dept_analysis = {}
        for emp in all_employees:
            dept = emp["department"]
            if dept not in dept_analysis:
                dept_analysis[dept] = {
                    "employee_count": 0,
                    "total_shifts": 0,
                    "total_payroll": 0,
                    "employees": []
                }
            
            dept_analysis[dept]["employee_count"] += 1
            dept_analysis[dept]["total_shifts"] += len(emp["shifts_worked"])
            dept_analysis[dept]["total_payroll"] += sum([c["amount_claimed"] for c in emp["salary_claims"]])
            dept_analysis[dept]["employees"].append(emp["name"])
        
        findings["department_analysis"] = dept_analysis
        analysis_log.append(f"  ✓ Analyzed {len(dept_analysis)} departments")
    
    return analysis_log, findings

def generate_intelligent_response(findings: Dict, query: str, sop_context: str) -> str:
    """Generate intelligent response based on database findings"""
    response_parts = []
    
    if "target_employee" in findings:
        emp = findings["target_employee"]
        response_parts.append(f"Employee Profile: {emp['name']} (ID: {emp['employee_id']}), {emp['designation']} in {emp['department']}")
    
    if "payroll_analysis" in findings:
        payroll = findings["payroll_analysis"]
        response_parts.append(f"Payroll Summary: {payroll['payroll_summary']}")
        if payroll.get("highest_earner"):
            top_earner = payroll["highest_earner"]
            response_parts.append(f"Top Earner: {top_earner['employee']} (₹{top_earner['monthly_earnings']:,})")
    
    if "attendance_analysis" in findings:
        attendance = findings["attendance_analysis"]
        response_parts.append(f"Attendance: {attendance['total_company_shifts']} total shifts, {attendance['total_company_hours']} hours")
        if attendance.get("best_attendance"):
            best = attendance["best_attendance"]
            response_parts.append(f"Best Attendance: {best['employee']} ({best['attendance_rate']}%)")
    
    if "claims_analysis" in findings:
        claims = findings["claims_analysis"]
        if claims["pending_claims"]:
            response_parts.append(f"Pending: {len(claims['pending_claims'])} claims worth ₹{claims['pending_amount']:,}")
    
    if "performance_analysis" in findings:
        perf = findings["performance_analysis"]
        if perf.get("top_performer"):
            top_perf = perf["top_performer"]
            response_parts.append(f"Top Performer: {top_perf['employee']} (Score: {top_perf['overall_score']:.1f})")
    
    if "employee_overview" in findings:
        overview = findings["employee_overview"]
        response_parts.append(f"Company Overview: {overview['total_employees']} employees across {len(overview['departments'])} departments")
    
    return " | ".join(response_parts) if response_parts else "Database analysis completed successfully"

@mcp.tool()
async def sop_retriever_tool(
    query: str,
    namespace: str = "test_namespace",
    doc_category: str = "test",
    topn: int = 5,
) -> dict:
    """
    Retrieve relevant SOP instructions to determine workflow and database access needs.
    """
    try:
        logger.info(f"📋 RETRIEVING SOP for query: '{query}'")
        
        docs = await retrieve_sop_docs(query, namespace, doc_category, topn)
        
        instructions = "\n\n".join([doc.text for doc in docs] if docs else ["Check employee records for payroll data"])
        
        analysis_result = analyze_sop_requirements(instructions, query)
        
        logger.info(f"🎯 SOP Analysis: Database access required = {analysis_result['requires_database_access']}")
        
        return {
            "sop_instructions": instructions,
            "requires_database_access": analysis_result["requires_database_access"],
            "database_indicators_found": analysis_result["database_indicators_found"],
            "user_query_requires_data": analysis_result["user_query_requires_data"],
            "sop_requires_data": analysis_result["sop_requires_data"],
            "retrieved_docs_count": len(docs),
            "guidance": "Proceed to database analysis if requires_database_access is True"
        }
        
    except Exception as e:
        logger.error(f"❌ SOP Retrieval Error: {str(e)}")
        return {
            "error": str(e),
            "sop_instructions": "Error retrieving SOP. Proceed with general assistance.",
            "requires_database_access": True
        }

@mcp.tool()
async def database_analyzer_tool(
    query: str,
    sop_context: str = "",
    analysis_type: str = "comprehensive"
) -> dict:
    """
    Comprehensive database analysis tool that intelligently searches and analyzes employee data.
    """
    try:
        logger.info(f"🗄️  ANALYZING DATABASE based on SOP requirements")
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"📋 SOP Context: {sop_context[:100]}..." if len(sop_context) > 100 else f"📋 SOP Context: {sop_context}")
        
        all_employees = user_database.get_all_employees()
        company_stats = user_database.get_company_stats()
        
        logger.info(f"→ Database loaded: {len(all_employees)} employees found")
        
        analysis_log, findings = perform_database_analysis(query, sop_context, all_employees, company_stats)
        
        response = generate_intelligent_response(findings, query, sop_context)
        
        action_items = []
        if "claims_analysis" in findings and findings["claims_analysis"]["pending_claims"]:
            pending_count = len(findings["claims_analysis"]["pending_claims"])
            pending_amount = findings["claims_analysis"]["pending_amount"]
            action_items.append(f"Review {pending_count} pending claims totaling ₹{pending_amount:,}")
        
        if "performance_analysis" in findings:
            safety_incidents = findings["performance_analysis"]["total_safety_incidents"]
            if safety_incidents > 0:
                action_items.append(f"Address {safety_incidents} safety incidents across workforce")
        
        analysis_log.append(f"✅ Database analysis complete - examined {len(findings)} data categories")
        
        return {
            "analysis_status": "completed",
            "database_accessed": True,
            "analysis_log": analysis_log,
            "findings": findings,
            "intelligent_response": response,
            "action_items": action_items,
            "employees_analyzed": len(all_employees),
            "data_categories": list(findings.keys()),
            "summary": f"Analyzed database with {len(all_employees)} employees across {len(findings)} data categories"
        }
        
    except Exception as e:
        logger.error(f"Database analysis error: {str(e)}")
        return {
            "analysis_status": "error",
            "error": str(e),
            "message": "Error during database analysis"
        }

@mcp.tool()
async def health_check() -> dict:
    """Health check endpoint for the MCP server"""
    return {
        "status": "healthy",
        "server": "SOP Database MCP Server",
        "timestamp": str(asyncio.get_event_loop().time()),
        "tools_available": ["sop_retriever_tool", "database_analyzer_tool", "health_check"]
    }

if __name__ == "__main__":
    mcp.run(transport="sse")