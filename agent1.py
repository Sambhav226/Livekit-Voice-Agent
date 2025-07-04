import os
import json
import time
import httpx
import traceback
import asyncio
import logging
import random
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, ChatContext, AgentSession, function_tool, RunContext, RoomInputOptions
from livekit.plugins import (
    openai,
    sarvam,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from VectorSearch import VectorSearch
from DynamicUserDatabase import DynamicUserDatabase

load_dotenv()

# --------------------------
# Initialize Retrieval Service
# --------------------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "your-index-name")
vector_service = VectorSearch(index_name=PINECONE_INDEX)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# Initialize the dynamic database
user_database = DynamicUserDatabase()

class SOPVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions='''
            You are an AI agent that follows Standard Operating Procedures (SOPs) and has access to employee database.
            
            WORKFLOW:
            1. For any user query, ALWAYS start by retrieving relevant SOPs using `sop_retriever_tool`
            2. Analyze the SOP response to determine if database access is required
            3. Only call `database_analyzer_tool` if the SOP indicates you need to:
               - Check employee records
               - Verify payroll/salary data
               - Analyze shift information
               - Review employee performance
               - Access any employee-related data
            4. Provide comprehensive responses based on SOP guidance and database findings
            
            Remember: SOP dictates when to access the database, not the user query directly.
            '''
        )

    @function_tool()
    async def sop_retriever_tool(
        self,
        context: RunContext,
        query: str,
        namespace: str = "test_namespace",
        doc_category: str = "test",
        topn: int = 5,
    ) -> dict[str, any]:
        """
        Retrieve relevant SOP instructions to determine workflow and database access needs.
        """
        try:
            print(f"📋 RETRIEVING SOP for query: '{query}'")
            
            docs = await vector_service.retrieval(
                query=query,
                namespace=namespace,
                doc_category=doc_category,
                rerank_threshold=0.7,
                topn=topn
            )
            
            instructions = "\n\n".join([doc.text for doc in docs])
            clean_instructions = instructions.replace('*', '').replace('#', '').replace('- ', '')
            clean_instructions = ' '.join(clean_instructions.split())
            
            # Analyze SOP for database access requirements
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
            
            # Also check user query for employee-specific requests
            employee_query_indicators = [
                "employee", "worker", "staff", "payroll", "salary", "wage", "shift",
                "earnings", "performance", "attendance", "profile", "record"
            ]
            
            user_requests_employee_data = any(
                indicator.lower() in query.lower() 
                for indicator in employee_query_indicators
            )
            
            # Final decision: SOP indicates OR user directly asks for employee data
            final_decision = requires_database_access or user_requests_employee_data
            
            found_indicators = [
                indicator for indicator in database_access_indicators 
                if indicator.lower() in clean_instructions.lower()
            ]
            
            print(f"🎯 SOP Analysis: Database access required = {final_decision}")
            
            return {
                "sop_instructions": clean_instructions,
                "requires_database_access": final_decision,
                "database_indicators_found": found_indicators,
                "user_query_requires_data": user_requests_employee_data,
                "sop_requires_data": requires_database_access,
                "retrieved_docs_count": len(docs),
                "guidance": "Proceed to database analysis if requires_database_access is True"
            }
            
        except Exception as e:
            print(f"❌ SOP Retrieval Error: {str(e)}")
            return {
                "error": str(e),
                "sop_instructions": "Error retrieving SOP. Proceed with general assistance.",
                "requires_database_access": True  # Default to allowing database access on error
            }

    @function_tool()
    async def database_analyzer_tool(
        self,
        context: RunContext,
        query: str,
        sop_context: str = "",
        analysis_type: str = "comprehensive"
    ) -> dict[str, Any]:
        """
        Comprehensive database analysis tool that intelligently searches and analyzes employee data.
        Called only when SOP indicates database access is required.
        """
        try:
            print(f"🗄️  ANALYZING DATABASE based on SOP requirements")
            print(f"📝 Query: '{query}'")
            print(f"📋 SOP Context: {sop_context[:100]}..." if len(sop_context) > 100 else f"📋 SOP Context: {sop_context}")
            
            analysis_log = ["Database analysis initiated per SOP requirements"]
            findings = {}
            
            # Get all available data
            all_employees = user_database.get_all_employees()
            company_stats = user_database.get_company_stats()
            
            analysis_log.append(f"→ Database loaded: {len(all_employees)} employees found")
            
            # Smart query analysis to determine what data to retrieve
            query_lower = query.lower()
            sop_lower = sop_context.lower() if sop_context else ""
            combined_context = f"{query_lower} {sop_lower}"
            
            # Employee identification
            if any(word in combined_context for word in ['employee', 'worker', 'staff', 'name', 'who', 'profile']):
                analysis_log.append("→ Employee identification analysis")
                
                # Check if specific employee mentioned
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
                    # Provide overview of all employees
                    findings["employee_overview"] = {
                        "total_employees": len(all_employees),
                        "employees_list": [{"id": emp["employee_id"], "name": emp["name"], "dept": emp["department"]} for emp in all_employees[:5]],
                        "departments": list(set([emp["department"] for emp in all_employees]))
                    }
                    analysis_log.append("  ✓ Employee overview generated")
            
            # Payroll and salary analysis
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
                    "highest_earner": max(wage_distribution, key=lambda x: x["monthly_earnings"]),
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
                    
                    # Analyze shift patterns
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
                    "average_shifts_per_employee": total_shifts / len(all_employees),
                    "best_attendance": max(attendance_records, key=lambda x: x["attendance_rate"])
                }
                analysis_log.append(f"  ✓ Analyzed {total_shifts} shifts totaling {total_hours} hours")
            
            # Performance analysis
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
                    "top_performer": max(performance_data, key=lambda x: x["overall_score"]),
                    "average_quality_rating": sum([p["quality_rating"] for p in performance_data]) / len(performance_data),
                    "total_safety_incidents": sum([p["safety_incidents"] for p in performance_data])
                }
                analysis_log.append("  ✓ Performance metrics analyzed for all employees")
            
            # Claims and pending payments analysis
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
            
            # Generate intelligent response based on findings
            response = self._generate_intelligent_response(findings, query, sop_context)
            
            # Generate action items
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
            return {
                "analysis_status": "error",
                "error": str(e),
                "message": "Error during database analysis"
            }
    
    def _generate_intelligent_response(self, findings: dict, query: str, sop_context: str) -> str:
        """Generate intelligent response based on database findings"""
        response_parts = []
        
        if "target_employee" in findings:
            emp = findings["target_employee"]
            response_parts.append(f"Employee Profile: {emp['name']} (ID: {emp['employee_id']}), {emp['designation']} in {emp['department']}")
        
        if "payroll_analysis" in findings:
            payroll = findings["payroll_analysis"]
            response_parts.append(f"Payroll Summary: {payroll['payroll_summary']}")
            if "highest_earner" in payroll:
                top_earner = payroll["highest_earner"]
                response_parts.append(f"Top Earner: {top_earner['employee']} (₹{top_earner['monthly_earnings']:,})")
        
        if "attendance_analysis" in findings:
            attendance = findings["attendance_analysis"]
            response_parts.append(f"Attendance: {attendance['total_company_shifts']} total shifts, {attendance['total_company_hours']} hours")
            best = attendance["best_attendance"]
            response_parts.append(f"Best Attendance: {best['employee']} ({best['attendance_rate']}%)")
        
        if "claims_analysis" in findings:
            claims = findings["claims_analysis"]
            if claims["pending_claims"]:
                response_parts.append(f"Pending: {len(claims['pending_claims'])} claims worth ₹{claims['pending_amount']:,}")
        
        if "performance_analysis" in findings:
            perf = findings["performance_analysis"]
            top_perf = perf["top_performer"]
            response_parts.append(f"Top Performer: {top_perf['employee']} (Score: {top_perf['overall_score']:.1f})")
        
        if "employee_overview" in findings:
            overview = findings["employee_overview"]
            response_parts.append(f"Company Overview: {overview['total_employees']} employees across {len(overview['departments'])} departments")
        
        return " | ".join(response_parts) if response_parts else "Database analysis completed successfully"

# --------------------------
# Entrypoint
# --------------------------
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    try:
        metadata_str = ctx.job.metadata or "{}"
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        metadata = {}

    user_name = metadata.get("user_name", "user")
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="manisha", api_key=SARVAM_API_KEY),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="assistant", 
        content=f"Hello {user_name}! I'm your AI assistant with access to employee database. I follow SOPs and can analyze comprehensive employee data including profiles, payroll, shifts, performance, and more. How can I help you today?"
    )

    await session.start(
        room=ctx.room,
        agent=SOPVoiceAgent(chat_ctx=chat_ctx),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))