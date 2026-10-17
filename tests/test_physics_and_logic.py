import pytest
import math

class TestPhysicsAndDomainLogic:
    def test_ttc_calculation(self):
        # Time to Collision: TTC = headway / (speed / 3.6)
        def calc_ttc(speed_kmh, headway_m):
            v_ms = speed_kmh / 3.6
            return headway_m / max(0.5, v_ms)

        # Standard case: 65 km/h (18.05 m/s), 28m headway -> ~1.55s
        ttc = calc_ttc(65.0, 28.0)
        assert 1.5 < ttc < 1.6

        # Zero speed case: should not divide by zero
        ttc_zero = calc_ttc(0.0, 10.0)
        assert ttc_zero == 20.0  # 10 / 0.5

        # Extreme high speed: 180 km/h (50 m/s), 5m headway -> 0.1s
        ttc_crit = calc_ttc(180.0, 5.0)
        assert ttc_crit < 0.2

    def test_stopping_distance_formula(self):
        # d_rxn = v_ms * t_rxn
        # d_skid = (v_ms ** 2) / (2 * g * max(0.06, mu + sin_theta))
        def calc_stopping(speed_kmh, rxn_sec, mu, abs_active, grade_pct):
            v_ms = speed_kmh / 3.6
            eff_mu = mu if abs_active else (mu * 0.72)
            grade_sin = grade_pct / 100.0
            denom = 2.0 * 9.81 * max(0.06, (eff_mu + grade_sin))
            d_rxn = v_ms * rxn_sec
            d_skid = (v_ms ** 2) / denom
            return d_rxn, d_skid, d_rxn + d_skid

        # Dry Bitumen (mu = 0.82), Alert driver (0.8s), ABS active, 0% slope, 65 km/h
        rxn, skid, total = calc_stopping(65.0, 0.8, 0.82, True, 0.0)
        assert 14.0 < rxn < 15.0  # ~14.4m
        assert 20.0 < skid < 21.0  # ~20.3m
        assert 34.0 < total < 36.0  # ~34.7m

        # Wet Asphalt without ABS: skid should be significantly longer
        _, skid_wet_noabs, total_wet = calc_stopping(65.0, 0.8, 0.52, False, 0.0)
        assert skid_wet_noabs > skid
        assert total_wet > total

        # Downhill slope (-10%): braking distance should increase
        _, skid_downhill, _ = calc_stopping(65.0, 0.8, 0.82, True, -10.0)
        assert skid_downhill > skid

        # Extreme steep downhill: denominator should be clamped by 0.06 to avoid zero div / negative
        _, skid_extreme, _ = calc_stopping(65.0, 0.8, 0.10, False, -20.0)
        assert skid_extreme > 0

    def test_pothole_shock_g_force(self):
        def calc_g_shock(approach_speed_kmh, cavity_depth_cm):
            v_ms = approach_speed_kmh / 3.6
            depth_m = cavity_depth_cm / 100.0
            # Shock load G-force: a / g = (v^2 * depth / (2 * stroke * g))
            suspension_stroke_m = 0.12
            g_force = (v_ms ** 2 * depth_m) / (2 * 9.81 * suspension_stroke_m)
            return g_force

        # 55 km/h approach, 8.5 cm pothole
        g = calc_g_shock(55.0, 8.5)
        assert 8.0 < g < 9.0  # Around 8.4 Gs (Severe suspension shock)

        # 20 km/h crawl, 3 cm pothole
        g_low = calc_g_shock(20.0, 3.0)
        assert g_low < 1.0  # Low shock

    def test_mva_challan_calculator(self):
        def calc_challan(riders, speed, limit, driver_helmet, pillion_helmet, cargo_overhang):
            items = []
            if not driver_helmet:
                items.append(("Section 129", 1000))
            if riders > 1 and not pillion_helmet:
                items.append(("Section 129(2)", 1000 * (riders - 1)))
            if riders >= 3:
                items.append(("Section 128", 2000))
            if speed > limit:
                items.append(("Section 183(1)", 2000))
            if cargo_overhang:
                items.append(("Section 190(2)", 1500))

            total_fine = sum(amt for _, amt in items)
            return items, total_fine

        # No violations
        items_clean, fine_clean = calc_challan(1, 45.0, 50.0, True, True, False)
        assert len(items_clean) == 0
        assert fine_clean == 0

        # Triple riding, no helmets for anyone, overspeeding (75 in 50), cargo overhang
        items_all, fine_all = calc_challan(3, 75.0, 50.0, False, False, True)
        assert len(items_all) == 5
        # Driver helmet: 1000, 2 pillions no helmet: 2000, triple riding: 2000, overspeed: 2000, cargo: 1500
        # Total = 8500
        assert fine_all == 8500

    def test_insurance_claim_settlement(self):
        def calc_insurance_claim(gross_estimate, tier, deductible, airbag_deployed, frame_twist, oem_parts):
            net_gross = gross_estimate
            if airbag_deployed:
                net_gross += 45000.0
            if frame_twist:
                net_gross += 65000.0
            if oem_parts:
                net_gross *= 1.15

            if "Zero-Depreciation" in tier:
                depr_rate = 0.0
                insured_rate = 1.0
            elif "Standard" in tier:
                depr_rate = 0.20
                insured_rate = 0.80
            else:  # Third party
                depr_rate = 1.0
                insured_rate = 0.0

            claimable_base = max(0.0, net_gross - deductible)
            insurer_payout = claimable_base * insured_rate
            customer_payable = net_gross - insurer_payout

            return net_gross, insurer_payout, customer_payable

        # Zero-dep policy: ₹50,000 base damage, ₹2,500 deductible, no airbag, no twist, standard parts
        gross, insurer, customer = calc_insurance_claim(50000.0, "Zero-Depreciation", 2500.0, False, False, False)
        assert gross == 50000.0
        assert insurer == 47500.0  # 50000 - 2500
        assert customer == 2500.0  # Deductible only

        # Third-party policy: Insurer pays 0 on own damage
        gross_tp, insurer_tp, customer_tp = calc_insurance_claim(50000.0, "Third-Party", 2500.0, False, False, False)
        assert insurer_tp == 0.0
        assert customer_tp == gross_tp
