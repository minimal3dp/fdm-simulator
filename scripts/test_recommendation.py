#!/usr/bin/env python3
"""
Test the /materials/recommend endpoint with the newly ingested data.
"""

import requests

API_BASE = "http://127.0.0.1:8000"


def test_basic_recommendation():
    """Test basic recommendation with balanced weights."""
    payload = {
        "constraints": {},
        "weight_strength": 1.0,
        "weight_temp": 1.0,
        "weight_cost": 1.0,
        "weight_density": 1.0,
    }

    response = requests.post(f"{API_BASE}/materials/recommend", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("✓ Basic Recommendation Test PASSED")
        print("  Top 3 materials:")
        for i, mat in enumerate(data["recommendations"][:3], 1):
            print(f"    {i}. {mat['name']} (score: {mat['score']})")
            if mat['missing_fields']:
                print(f"       Missing: {', '.join(mat['missing_fields'])}")
        return True
    else:
        print(f"✗ Basic Recommendation Test FAILED: {response.status_code}")
        print(f"  {response.text}")
        return False


def test_strength_focused():
    """Test recommendation focused on strength."""
    payload = {
        "constraints": {},
        "weight_strength": 5.0,
        "weight_temp": 1.0,
        "weight_cost": 0.5,
        "weight_density": 0.5,
    }

    response = requests.post(f"{API_BASE}/materials/recommend", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Strength-Focused Test PASSED")
        print("  Top 3 materials for high strength:")
        for i, mat in enumerate(data["recommendations"][:3], 1):
            print(f"    {i}. {mat['name']} (score: {mat['score']})")
        return True
    else:
        print(f"\n✗ Strength-Focused Test FAILED: {response.status_code}")
        return False


def test_with_constraints():
    """Test recommendation with minimum strength constraint."""
    payload = {
        "constraints": {"min_strength_MPa": 50.0, "max_temp_C": 100.0},
        "weight_strength": 1.0,
        "weight_temp": 1.0,
        "weight_cost": 1.0,
        "weight_density": 1.0,
    }

    response = requests.post(f"{API_BASE}/materials/recommend", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Constraint Test PASSED")
        print("  Materials with strength ≥50 MPa and temp ≤100°C:")
        for i, mat in enumerate(data["recommendations"][:5], 1):
            print(f"    {i}. {mat['name']} (score: {mat['score']})")
        return True
    else:
        print(f"\n✗ Constraint Test FAILED: {response.status_code}")
        return False


def main():
    """Run all tests."""
    print("Testing Materials Recommendation Endpoint")
    print("=" * 50)

    try:
        # Check if server is running
        response = requests.get(f"{API_BASE}/materials")
        if response.status_code != 200:
            print("✗ Server not responding. Start with: python run_all_training.py")
            return

        print(f"✓ Server running. Materials available: {len(response.json())}\n")

        # Run tests
        results = []
        results.append(test_basic_recommendation())
        results.append(test_strength_focused())
        results.append(test_with_constraints())

        print("\n" + "=" * 50)
        print(f"Summary: {sum(results)}/{len(results)} tests passed")

    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Start with:")
        print("  python run_all_training.py")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
