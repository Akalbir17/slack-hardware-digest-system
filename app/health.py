"""
Health check module for Docker containers
Provides comprehensive health checking for all system components
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import httpx
except ImportError:
    httpx = None


class HealthChecker:
    """Comprehensive health checker for all system components"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "")
        self.redis_url = os.getenv("REDIS_URL", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity"""
        try:
            if not self.database_url:
                return {"status": "error", "message": "DATABASE_URL not configured"}
            
            if not asyncpg:
                return {"status": "error", "message": "asyncpg not installed"}
            
            conn = await asyncpg.connect(self.database_url)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            
            # Test table existence
            tables_query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            tables = await conn.fetch(tables_query)
            table_names = [row['table_name'] for row in tables]
            
            await conn.close()
            
            return {
                "status": "healthy",
                "message": "Database connection successful",
                "details": {
                    "query_result": result,
                    "tables_count": len(table_names),
                    "tables": table_names[:5]  # Show first 5 tables
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database connection failed: {str(e)}"
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            if not self.redis_url:
                return {"status": "error", "message": "REDIS_URL not configured"}
            
            if not redis:
                return {"status": "error", "message": "redis not installed"}
            
            redis_client = redis.from_url(self.redis_url)
            
            # Test ping
            pong = await redis_client.ping()
            
            # Test set/get
            test_key = f"health_check_{datetime.utcnow().timestamp()}"
            await redis_client.set(test_key, "test_value", ex=10)
            test_value = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            # Get Redis info
            info = await redis_client.info()
            
            await redis_client.close()
            
            return {
                "status": "healthy",
                "message": "Redis connection successful",
                "details": {
                    "ping": pong,
                    "test_operation": test_value.decode() if test_value else None,
                    "redis_version": info.get("redis_version"),
                    "memory_usage": info.get("used_memory_human")
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Redis connection failed: {str(e)}"
            }
    
    def check_openai(self) -> Dict[str, Any]:
        """Check OpenAI API configuration"""
        try:
            if not self.openai_api_key or self.openai_api_key == "your-key-here":
                return {
                    "status": "warning",
                    "message": "OpenAI API key not configured"
                }
            
            # Test API key format
            if not self.openai_api_key.startswith("sk-"):
                return {
                    "status": "error",
                    "message": "Invalid OpenAI API key format"
                }
            
            return {
                "status": "configured",
                "message": "OpenAI API key configured"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"OpenAI check failed: {str(e)}"
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            
            # Convert to GB
            total_gb = total // (1024**3)
            used_gb = used // (1024**3)
            free_gb = free // (1024**3)
            usage_percent = (used / total) * 100
            
            status = "healthy"
            if usage_percent > 90:
                status = "critical"
            elif usage_percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "message": f"Disk usage: {usage_percent:.1f}%",
                "details": {
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "usage_percent": round(usage_percent, 1)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Disk space check failed: {str(e)}"
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = datetime.utcnow()
        
        # Run async checks
        database_check = await self.check_database()
        redis_check = await self.check_redis()
        
        # Run sync checks
        openai_check = self.check_openai()
        disk_check = self.check_disk_space()
        
        # Determine overall health status
        overall_status = "healthy"
        critical_issues = []
        warnings = []
        
        for check_name, check_result in [
            ("database", database_check),
            ("redis", redis_check),
            ("openai", openai_check),
            ("disk", disk_check)
        ]:
            if check_result.get("status") == "error":
                critical_issues.append(f"{check_name}: {check_result.get('message')}")
                overall_status = "unhealthy"
            elif check_result.get("status") == "critical":
                critical_issues.append(f"{check_name}: {check_result.get('message')}")
                overall_status = "unhealthy"
            elif check_result.get("status") == "warning":
                warnings.append(f"{check_name}: {check_result.get('message')}")
                if overall_status == "healthy":
                    overall_status = "degraded"
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "status": overall_status,
            "timestamp": start_time.isoformat(),
            "duration_seconds": round(duration, 3),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            "checks": {
                "database": database_check,
                "redis": redis_check,
                "openai": openai_check,
                "disk": disk_check
            },
            "summary": {
                "critical_issues": critical_issues,
                "warnings": warnings,
                "total_checks": 4,
                "healthy_checks": sum(1 for c in [database_check, redis_check, openai_check, disk_check] 
                                    if c.get("status") in ["healthy", "configured"])
            }
        }


async def main():
    """Main function for command-line health checking"""
    checker = HealthChecker()
    result = await checker.run_all_checks()
    
    print(f"Health Check Result: {result['status'].upper()}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Duration: {result['duration_seconds']}s")
    print()
    
    for check_name, check_result in result['checks'].items():
        status = check_result['status']
        message = check_result['message']
        emoji = {
            'healthy': '✅',
            'configured': '⚙️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'degraded': '⚠️'
        }.get(status, '❓')
        
        print(f"{emoji} {check_name.title()}: {message}")
    
    print()
    summary = result['summary']
    print(f"Summary: {summary['healthy_checks']}/{summary['total_checks']} checks passed")
    
    if summary['critical_issues']:
        print("Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"  • {issue}")
    
    if summary['warnings']:
        print("Warnings:")
        for warning in summary['warnings']:
            print(f"  • {warning}")
    
    # Exit with appropriate code
    if result['status'] == 'unhealthy':
        sys.exit(1)
    elif result['status'] == 'degraded':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main()) 