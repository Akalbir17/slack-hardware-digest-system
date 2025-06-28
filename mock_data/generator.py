"""
Mock Slack Message Generator for Hardware Teams
Generates realistic hardware team communications including component shortages,
quality issues, timeline updates, and supply chain alerts.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import json


class MessageType(Enum):
    ALERT = "alert"
    UPDATE = "update" 
    QUESTION = "question"


class UrgencyLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MessageCategory(Enum):
    COMPONENT_SHORTAGE = "component_shortage"
    QUALITY_ISSUE = "quality_issue"
    TIMELINE_UPDATE = "timeline_update"
    SUPPLY_CHAIN_ALERT = "supply_chain_alert"
    TESTING_RESULTS = "testing_results"
    VENDOR_UPDATE = "vendor_update"
    MANUFACTURING_ISSUE = "manufacturing_issue"


@dataclass
class SlackMessage:
    """Represents a Slack message with metadata"""
    channel: str
    user: str
    content: str
    timestamp: datetime
    message_type: MessageType
    urgency: UrgencyLevel
    category: MessageCategory
    mentions: List[str] = None
    reactions: List[str] = None
    thread_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "channel": self.channel,
            "user": self.user,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "urgency": self.urgency.value,
            "category": self.category.value,
            "mentions": self.mentions or [],
            "reactions": self.reactions or [],
            "thread_count": self.thread_count
        }


class HardwareTeamMockGenerator:
    """Generates realistic hardware team Slack messages"""
    
    def __init__(self):
        self.users = [
            "alice.chen", "bob.martinez", "charlie.kim", "diana.patel", "eve.johnson",
            "frank.liu", "grace.wilson", "henry.yamamoto", "iris.singh", "jack.brown",
            "kelly.garcia", "liam.davis", "maya.rodriguez", "noah.thompson", "olivia.lee"
        ]
        
        self.channels = [
            "#hardware-gtm", "#engineering", "#supply-chain", "#quality-assurance",
            "#manufacturing", "#vendors", "#product-updates", "#alerts", "#general"
        ]
        
        # Hardware-specific data
        self.components = {
            "displays": [
                "AMOLED 6.1\" Samsung E7", "IPS LCD 5.5\" BOE NT35596", "OLED 6.8\" LG LP068WF1",
                "e-Paper 13.3\" E Ink Carta", "Mini-LED 12.9\" Sharp LQ120F1LH01",
                "LTPS LCD 10.1\" JDI LPM101M481A", "Flexible OLED 7.2\" Samsung Y-OCTA"
            ],
            "chipsets": [
                "Snapdragon 8 Gen 3", "MediaTek Dimensity 9300", "Apple A17 Pro", 
                "Exynos 2400", "Google Tensor G4", "Qualcomm QCM6490", "Unisoc Tiger T820"
            ],
            "batteries": [
                "Li-ion 4500mAh 3.85V", "LiFePO4 6000mAh 3.2V", "Li-Po 3200mAh 3.7V",
                "Solid-state 5000mAh 3.8V", "Graphene 4800mAh 3.9V"
            ],
            "sensors": [
                "IMU BMI088", "Gyroscope ICM-20948", "Accelerometer ADXL345",
                "Magnetometer AK09918C", "Pressure BMP390", "Proximity APDS-9960"
            ],
            "memory": [
                "LPDDR5 12GB Samsung", "UFS 3.1 256GB Micron", "eMMC 5.1 64GB SanDisk",
                "NAND Flash 1TB SK Hynix", "SRAM 8MB Cypress"
            ]
        }
        
        self.vendors = [
            "Foxconn", "Pegatron", "Wistron", "Compal", "Quanta", "Inventec",
            "Flex", "Jabil", "Sanmina", "Celestica", "Benchmark Electronics",
            "Hon Hai", "New Kinpo Group", "Cal-Comp Electronics"
        ]
        
        self.project_codenames = [
            "Phoenix", "Aurora", "Titan", "Nova", "Vega", "Orion", "Atlas",
            "Nexus", "Quantum", "Eclipse", "Helios", "Zenith", "Prism", "Vector"
        ]
        
        self.milestones = ["EVT", "EVT2", "DVT", "DVT2", "PVT", "MP"]
        
        # Message templates
        self.shortage_templates = [
            "🚨 CRITICAL: {component} shortage at {vendor}. Current inventory: {quantity} units, need {required} for {milestone}",
            "⚠️ Supply constraint on {component}. Lead time extended to {weeks} weeks. Impact on {project} timeline",
            "📊 Component availability update: {component} showing {availability}% allocation for Q{quarter}",
            "🔄 Alternative sourcing needed for {component}. Primary supplier {vendor} experiencing {issue}",
            "📈 Price increase alert: {component} cost up {percentage}% due to {reason}"
        ]
        
        self.quality_templates = [
            "📉 Quality alert: {component} yield rate dropped to {yield_rate}% (target: {target}%)",
            "🔍 DPPM update: {component} showing {dppm} DPPM in {milestone} build",
            "⚡ Functional test results: {pass_rate}% pass rate on {component} ({total} units tested)",
            "🌡️ Thermal testing: {component} exceeding {temp}°C under {condition} conditions",
            "🔋 Battery life impact: {reduction}% decrease due to {component} power consumption"
        ]
        
        self.timeline_templates = [
            "📅 {milestone} milestone update for {project}: {status} - {details}",
            "⏰ Schedule revision: {project} {milestone} moved from {old_date} to {new_date}",
            "✅ {milestone} exit criteria met for {project}. Moving to {next_milestone}",
            "🚦 Gating item: {issue} blocking {project} {milestone} completion",
            "📊 Program status: {project} is {status} by {weeks} weeks due to {reason}"
        ]
        
        self.supply_chain_templates = [
            "🌏 Geopolitical impact: {vendor} operations affected by {event}",
            "🚢 Logistics update: {delay} day delay on {component} shipment from {location}",
            "💰 Cost impact: {category} components seeing {percentage}% inflation due to {factor}",
            "🏭 Manufacturing capacity: {vendor} expanding {component} production by {percentage}%",
            "📦 Inventory optimization: Suggesting {weeks} weeks safety stock for {component}"
        ]
        
        self.testing_templates = [
            "🧪 Reliability testing: {component} completed {cycles} cycles, {failures} failures",
            "📏 Dimensional analysis: {component} showing {variance}mm variance in {measurement}",
            "⚡ EMI/EMC testing: {result} on {component} at {frequency}MHz",
            "🌊 Environmental testing: {component} {result} at {temp}°C, {humidity}% RH",
            "🔌 Power consumption: {component} drawing {power}mW ({percentage}% over spec)"
        ]
        
        self.question_templates = [
            "❓ Has anyone seen similar {issue} with {component} from {vendor}?",
            "🤔 What's our contingency for {component} if {vendor} can't deliver?",
            "💭 Should we consider alternative {component} for {project} given {concern}?",
            "🔍 Can {user} share the latest {test_type} results for {component}?",
            "📋 Do we have approval to proceed with {milestone} despite {issue}?"
        ]

    def _random_component(self, category: str = None) -> str:
        """Get random component, optionally from specific category"""
        if category and category in self.components:
            return random.choice(self.components[category])
        
        all_components = []
        for comp_list in self.components.values():
            all_components.extend(comp_list)
        return random.choice(all_components)

    def _random_vendor(self) -> str:
        """Get random vendor name"""
        return random.choice(self.vendors)

    def _random_project(self) -> str:
        """Get random project codename"""
        return random.choice(self.project_codenames)

    def _random_milestone(self) -> str:
        """Get random milestone"""
        return random.choice(self.milestones)

    def _random_user(self) -> str:
        """Get random user"""
        return random.choice(self.users)

    def _random_channel(self) -> str:
        """Get random channel"""
        return random.choice(self.channels)

    def _generate_shortage_message(self) -> SlackMessage:
        """Generate component shortage message"""
        template = random.choice(self.shortage_templates)
        urgency = random.choice([UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.NORMAL])
        
        # Generate realistic data
        component = self._random_component()
        vendor = self._random_vendor()
        quantity = random.randint(50, 500)
        required = random.randint(quantity + 100, quantity + 1000)
        weeks = random.randint(4, 16)
        availability = random.randint(20, 80)
        quarter = random.randint(1, 4)
        percentage = random.randint(10, 40)
        
        issues = ["supply chain disruption", "factory closure", "material shortage", "quality hold", "logistics delay"]
        reasons = ["raw material cost increase", "supply chain constraints", "regulatory changes", "market demand"]
        
        content = template.format(
            component=component,
            vendor=vendor,
            quantity=quantity,
            required=required,
            milestone=self._random_milestone(),
            weeks=weeks,
            project=self._random_project(),
            availability=availability,
            quarter=quarter,
            issue=random.choice(issues),
            percentage=percentage,
            reason=random.choice(reasons)
        )
        
        return SlackMessage(
            channel=random.choice(["#supply-chain", "#alerts", "#hardware-gtm"]),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 60)),
            message_type=MessageType.ALERT,
            urgency=urgency,
            category=MessageCategory.COMPONENT_SHORTAGE,
            reactions=random.choices(["😟", "🚨", "👀", "📊"], k=random.randint(0, 3))
        )

    def _generate_quality_message(self) -> SlackMessage:
        """Generate quality issue message"""
        template = random.choice(self.quality_templates)
        urgency = random.choice([UrgencyLevel.HIGH, UrgencyLevel.NORMAL, UrgencyLevel.LOW])
        
        # Generate realistic quality data
        component = self._random_component()
        yield_rate = round(random.uniform(85.0, 99.5), 1)
        target_yield = round(yield_rate + random.uniform(1.0, 5.0), 1)
        dppm = random.randint(50, 2000)
        pass_rate = round(random.uniform(88.0, 99.8), 1)
        total_units = random.randint(100, 5000)
        temp = random.randint(65, 95)
        reduction = random.randint(5, 25)
        
        conditions = ["stress testing", "automotive qualification", "extended operation", "high humidity"]
        
        content = template.format(
            component=component,
            yield_rate=yield_rate,
            target=target_yield,
            dppm=dppm,
            milestone=self._random_milestone(),
            pass_rate=pass_rate,
            total=total_units,
            temp=temp,
            condition=random.choice(conditions),
            reduction=reduction
        )
        
        return SlackMessage(
            channel=random.choice(["#quality-assurance", "#engineering", "#manufacturing"]),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 120)),
            message_type=MessageType.UPDATE,
            urgency=urgency,
            category=MessageCategory.QUALITY_ISSUE,
            reactions=random.choices(["📊", "🔍", "⚠️", "✅"], k=random.randint(0, 2))
        )

    def _generate_timeline_message(self) -> SlackMessage:
        """Generate timeline update message"""
        template = random.choice(self.timeline_templates)
        urgency = random.choice([UrgencyLevel.NORMAL, UrgencyLevel.HIGH])
        
        # Generate timeline data
        milestone = self._random_milestone()
        project = self._random_project()
        
        statuses = ["on track", "delayed", "ahead of schedule", "at risk"]
        status = random.choice(statuses)
        
        details = [
            "all exit criteria met", "waiting for component qualification",
            "pending regulatory approval", "supply chain dependency",
            "additional testing required", "design optimization in progress"
        ]
        
        # Generate dates
        base_date = datetime.now() + timedelta(days=random.randint(7, 90))
        old_date = base_date.strftime("%m/%d")
        new_date = (base_date + timedelta(days=random.randint(7, 21))).strftime("%m/%d")
        
        next_milestones = {"EVT": "DVT", "DVT": "PVT", "PVT": "MP"}
        next_milestone = next_milestones.get(milestone, "Production")
        
        issues = ["component shortage", "test failure", "supplier issue", "design change"]
        reasons = ["component delays", "test failures", "supplier issues", "design changes"]
        
        content = template.format(
            milestone=milestone,
            project=project,
            status=status,
            details=random.choice(details),
            old_date=old_date,
            new_date=new_date,
            next_milestone=next_milestone,
            issue=random.choice(issues),
            weeks=random.randint(1, 8),
            reason=random.choice(reasons)
        )
        
        return SlackMessage(
            channel=random.choice(["#hardware-gtm", "#product-updates", "#engineering"]),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 180)),
            message_type=MessageType.UPDATE,
            urgency=urgency,
            category=MessageCategory.TIMELINE_UPDATE,
            mentions=random.choices(self.users, k=random.randint(0, 2)),
            reactions=random.choices(["📅", "⏰", "✅", "🚦"], k=random.randint(0, 2))
        )

    def _generate_supply_chain_message(self) -> SlackMessage:
        """Generate supply chain alert message"""
        template = random.choice(self.supply_chain_templates)
        urgency = random.choice([UrgencyLevel.HIGH, UrgencyLevel.NORMAL])
        
        vendor = self._random_vendor()
        component = self._random_component()
        delay = random.randint(3, 14)
        percentage = random.randint(10, 50)
        weeks = random.randint(2, 12)
        cycles = random.randint(500, 5000)
        failures = random.randint(0, 20)
        
        events = ["trade regulations", "natural disaster", "port congestion", "labor strike"]
        locations = ["Shenzhen", "Taiwan", "Singapore", "Vietnam", "Malaysia"]
        factors = ["energy costs", "raw material shortage", "transportation costs"]
        categories = ["semiconductor", "display", "battery", "sensor"]
        
        content = template.format(
            vendor=vendor,
            event=random.choice(events),
            delay=delay,
            component=component,
            location=random.choice(locations),
            category=random.choice(categories),
            percentage=percentage,
            factor=random.choice(factors),
            weeks=weeks
        )
        
        return SlackMessage(
            channel=random.choice(["#supply-chain", "#alerts", "#vendors"]),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 240)),
            message_type=MessageType.ALERT,
            urgency=urgency,
            category=MessageCategory.SUPPLY_CHAIN_ALERT,
            reactions=random.choices(["🌏", "🚢", "💰", "📦"], k=random.randint(0, 2))
        )

    def _generate_testing_message(self) -> SlackMessage:
        """Generate testing results message"""
        template = random.choice(self.testing_templates)
        urgency = UrgencyLevel.NORMAL
        
        component = self._random_component()
        cycles = random.randint(1000, 10000)
        failures = random.randint(0, 50)
        variance = round(random.uniform(0.1, 2.0), 2)
        frequency = random.choice([433, 868, 915, 2400, 5800])
        temp = random.randint(-20, 85)
        humidity = random.randint(10, 95)
        power = round(random.uniform(50.0, 500.0), 1)
        percentage = random.randint(5, 25)
        
        results = ["PASS", "FAIL", "MARGINAL"]
        measurements = ["length", "width", "thickness", "diameter"]
        test_types = ["functional", "reliability", "environmental", "EMI/EMC"]
        
        content = template.format(
            component=component,
            cycles=cycles,
            failures=failures,
            variance=variance,
            measurement=random.choice(measurements),
            result=random.choice(results),
            frequency=frequency,
            temp=temp,
            humidity=humidity,
            power=power,
            percentage=percentage
        )
        
        return SlackMessage(
            channel=random.choice(["#quality-assurance", "#engineering", "#testing"]),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 300)),
            message_type=MessageType.UPDATE,
            urgency=urgency,
            category=MessageCategory.TESTING_RESULTS,
            reactions=random.choices(["🧪", "📏", "⚡", "🔌"], k=random.randint(0, 2))
        )

    def _generate_question_message(self) -> SlackMessage:
        """Generate question message"""
        template = random.choice(self.question_templates)
        urgency = random.choice([UrgencyLevel.NORMAL, UrgencyLevel.LOW])
        
        component = self._random_component()
        vendor = self._random_vendor()
        project = self._random_project()
        milestone = self._random_milestone()
        user = self._random_user()
        
        issues = ["yield drop", "thermal issue", "power consumption", "mechanical fit"]
        concerns = ["cost increase", "supply risk", "quality issues", "timeline impact"]
        test_types = ["reliability", "functional", "environmental", "regulatory"]
        
        content = template.format(
            issue=random.choice(issues),
            component=component,
            vendor=vendor,
            project=project,
            concern=random.choice(concerns),
            user=user,
            test_type=random.choice(test_types),
            milestone=milestone
        )
        
        return SlackMessage(
            channel=self._random_channel(),
            user=self._random_user(),
            content=content,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 480)),
            message_type=MessageType.QUESTION,
            urgency=urgency,
            category=random.choice(list(MessageCategory)),
            thread_count=random.randint(0, 8),
            reactions=random.choices(["❓", "🤔", "💭", "👀"], k=random.randint(0, 3))
        )

    def generate_message(self, category: MessageCategory = None) -> SlackMessage:
        """Generate a single message, optionally of specific category"""
        if category:
            generators = {
                MessageCategory.COMPONENT_SHORTAGE: self._generate_shortage_message,
                MessageCategory.QUALITY_ISSUE: self._generate_quality_message,
                MessageCategory.TIMELINE_UPDATE: self._generate_timeline_message,
                MessageCategory.SUPPLY_CHAIN_ALERT: self._generate_supply_chain_message,
                MessageCategory.TESTING_RESULTS: self._generate_testing_message,
            }
            if category in generators:
                return generators[category]()
        
        # Random category
        generators = [
            self._generate_shortage_message,
            self._generate_quality_message,
            self._generate_timeline_message,
            self._generate_supply_chain_message,
            self._generate_testing_message,
            self._generate_question_message
        ]
        
        return random.choice(generators)()

    def generate_batch(self, count: int, category: MessageCategory = None) -> List[SlackMessage]:
        """Generate a batch of messages"""
        return [self.generate_message(category) for _ in range(count)]

    def generate_stream(self, rate_per_minute: float = 1.0, duration_minutes: int = 60) -> Generator[SlackMessage, None, None]:
        """Generate continuous stream of messages"""
        total_messages = int(rate_per_minute * duration_minutes)
        interval = 60.0 / rate_per_minute if rate_per_minute > 0 else 60.0
        
        for i in range(total_messages):
            yield self.generate_message()
            if i < total_messages - 1:  # Don't sleep after last message
                time.sleep(interval)

    def generate_daily_scenario(self, day_type: str = "normal") -> List[SlackMessage]:
        """Generate a full day scenario with realistic message patterns"""
        messages = []
        
        if day_type == "crisis":
            # Crisis day: many urgent messages
            messages.extend(self.generate_batch(5, MessageCategory.COMPONENT_SHORTAGE))
            messages.extend(self.generate_batch(3, MessageCategory.SUPPLY_CHAIN_ALERT))
            messages.extend(self.generate_batch(4, MessageCategory.QUALITY_ISSUE))
            messages.extend(self.generate_batch(2, MessageCategory.TIMELINE_UPDATE))
        elif day_type == "milestone":
            # Milestone day: focus on timeline and testing
            messages.extend(self.generate_batch(6, MessageCategory.TIMELINE_UPDATE))
            messages.extend(self.generate_batch(4, MessageCategory.TESTING_RESULTS))
            messages.extend(self.generate_batch(2, MessageCategory.QUALITY_ISSUE))
        else:
            # Normal day: mixed messages
            messages.extend(self.generate_batch(3, MessageCategory.COMPONENT_SHORTAGE))
            messages.extend(self.generate_batch(4, MessageCategory.QUALITY_ISSUE))
            messages.extend(self.generate_batch(3, MessageCategory.TIMELINE_UPDATE))
            messages.extend(self.generate_batch(2, MessageCategory.SUPPLY_CHAIN_ALERT))
            messages.extend(self.generate_batch(3, MessageCategory.TESTING_RESULTS))
            messages.extend(self.generate_batch(5))  # Random questions/updates
        
        # Sort by timestamp
        messages.sort(key=lambda x: x.timestamp)
        return messages

    def export_to_json(self, messages: List[SlackMessage], filename: str):
        """Export messages to JSON file"""
        data = [msg.to_dict() for msg in messages]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """Example usage of the generator"""
    generator = HardwareTeamMockGenerator()
    
    print("🤖 Hardware Team Mock Message Generator")
    print("=" * 50)
    
    # Generate some example messages
    print("\n📊 Sample Messages:")
    for i in range(5):
        msg = generator.generate_message()
        print(f"\n[{msg.urgency.value.upper()}] #{msg.channel[1:]} - {msg.user}")
        print(f"📝 {msg.content}")
        print(f"🏷️ Category: {msg.category.value} | Type: {msg.message_type.value}")
    
    # Generate daily scenario
    print("\n\n📅 Daily Crisis Scenario:")
    crisis_messages = generator.generate_daily_scenario("crisis")
    print(f"Generated {len(crisis_messages)} messages for crisis day")
    
    # Export example
    generator.export_to_json(crisis_messages, "crisis_day_messages.json")
    print("💾 Exported to crisis_day_messages.json")


if __name__ == "__main__":
    main() 