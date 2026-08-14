import os
import sys
import time
import requests
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000/predict/ad-click")


def run_ip_velocity_stress_test(test_ip: int = 88888, total_clicks: int = 150):
    """
    Stress-tests the real-time serving engine by sending a continuous stream
    from a single IP address to observe dynamic velocity escalation:
    1. Initial clicks (1-15): Benign behavior -> LOW_ALLOW
    2. Mid-rate clicks (16-60): Velocity threshold crossed -> MEDIUM_CHALLENGE
    3. High-velocity flooding (60+): Click flooding detected -> HIGH_BLOCK
    """
    print("======================================================================")
    print(f"🛡️ IP VELOCITY ESCALATION STRESS TEST: IP={test_ip}")
    print("======================================================================\n")
    print(f"🎯 Target Endpoint: {API_URL}")
    print(f"📊 Simulating a benign user turning into a continuous high-rate click flood ({total_clicks} clicks)...\n")

    low_count = 0
    challenge_count = 0
    block_count = 0

    print(f"{'Click #':<9} | {'IP':<7} | {'Risk Score':<12} | {'Tier':<18} | {'Action':<15} | {'Latency':<8}")
    print("-" * 80)

    for i in range(1, total_clicks + 1):
        payload = {
            "ip": test_ip,
            "app": 3,
            "device": 1,
            "os": 19,
            "channel": 379,
            "click_time": datetime.utcnow().isoformat(),
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                risk = data["fraud_probability"]
                tier = data["risk_tier"]
                latency = data["inference_latency_ms"]

                if tier == "LOW_ALLOW":
                    action = "🟢 ALLOW"
                    low_count += 1
                elif tier == "MEDIUM_CHALLENGE":
                    action = "⚠️ CHALLENGE (CAPTCHA)"
                    challenge_count += 1
                else:
                    action = "🛑 BLOCK (BLOCKED)"
                    block_count += 1

                # Print key milestones and sampled steps
                if i <= 10 or i % 15 == 0 or i == total_clicks or tier != "LOW_ALLOW":
                    print(f"#{i:03d}      | {data['ip']:<7} | {risk:<12.4f} | {tier:<18} | {action:<15} | {latency:.1f}ms")

            time.sleep(0.02)  # High speed flood
        except Exception as e:
            print(f"Connection error at click #{i}: {e}")
            break

    print("-" * 80)
    print(f"\n📊 Summary of Mitigation for IP {test_ip}:")
    print(f"   🟢 Normal / Allowed Clicks:    {low_count} clicks")
    print(f"   ⚠️ Challenged Clicks:          {challenge_count} clicks")
    print(f"   🛑 High-Risk Blocked Clicks:   {block_count} clicks")
    print("\n👉 Check Prometheus/Grafana (http://localhost:3000) to see the live gauge spike!")


if __name__ == "__main__":
    run_ip_velocity_stress_test()
