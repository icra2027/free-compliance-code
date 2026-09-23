#include <gtest/gtest.h>

#include "variable_impedance_controllers/utils/energy_tank.hpp"

TEST(EnergyTankTest, ConstructorSetsInitialEnergy) {
  EnergyTank tank(5.0, 0.0, 10.0);
  EXPECT_DOUBLE_EQ(tank.energy(), 5.0);
}

TEST(EnergyTankTest, AccumulatePositivePowerIncreasesEnergy) {
  EnergyTank tank(5.0, 0.0, 10.0);
  tank.accumulate_dissipated_power(2.0, 0.5);  // +1.0 J
  EXPECT_DOUBLE_EQ(tank.energy(), 6.0);
}

TEST(EnergyTankTest, AccumulateClampsAtEMax) {
  EnergyTank tank(9.0, 0.0, 10.0);
  // Repeatedly bank far more energy than the ceiling allows -- excess must simply not be
  // banked, not overflow or error.
  for (int i = 0; i < 100; ++i) {
    tank.accumulate_dissipated_power(1000.0, 1.0);
  }
  EXPECT_DOUBLE_EQ(tank.energy(), 10.0);
}

TEST(EnergyTankTest, AccumulateClampsAtEMin) {
  EnergyTank tank(1.0, 0.0, 10.0);
  // A large negative power (energy flowing OUT of the tank faster than it has any) must
  // clamp at the floor rather than go negative.
  tank.accumulate_dissipated_power(-1000.0, 1.0);
  EXPECT_DOUBLE_EQ(tank.energy(), 0.0);
}

TEST(EnergyTankTest, DepletedReflectsFloorState) {
  EnergyTank tank(1.0, 0.5, 10.0);
  EXPECT_FALSE(tank.depleted());

  tank.accumulate_dissipated_power(-100.0, 1.0);  // drive down to the floor
  EXPECT_TRUE(tank.depleted());
  EXPECT_DOUBLE_EQ(tank.energy(), 0.5);
}

TEST(EnergyTankTest, TryWithdrawSucceedsWhenEnergyAvailable) {
  EnergyTank tank(5.0, 0.0, 10.0);
  EXPECT_TRUE(tank.try_withdraw(3.0));
  EXPECT_DOUBLE_EQ(tank.energy(), 2.0);
}

// A withdrawal that would push the tank below its floor must be rejected AND leave the tank
// completely untouched -- the caller relies on this to decide whether to reject/clamp a
// stiffness increase without the tank's state being corrupted by the failed attempt.
TEST(EnergyTankTest, TryWithdrawFailsAndLeavesEnergyUnchangedWhenInsufficient) {
  EnergyTank tank(1.0, 0.5, 10.0);
  EXPECT_FALSE(tank.try_withdraw(0.9));  // would land at 0.1, below the 0.5 floor
  EXPECT_DOUBLE_EQ(tank.energy(), 1.0);
}

// Regression coverage for a real bug found on real hardware in fr3_bilateral_teleop (see
// energy_tank.hpp's kFloorEpsilon comment): a withdrawal landing EXACTLY on the floor was
// spuriously rejected by a strict `<` comparison, because `1.0 - 0.9` is not exactly
// representable in double precision. Without the epsilon tolerance, this case fails.
TEST(EnergyTankTest, TryWithdrawExactlyToFloorSucceedsDespiteFloatingPointRounding) {
  EnergyTank tank(1.0, 0.1, 10.0);
  ASSERT_NE(1.0 - 0.9, 0.1) << "this test relies on double-precision rounding of 1.0 - 0.9; "
    "if this assertion fires the test needs a different repro value";

  EXPECT_TRUE(tank.try_withdraw(0.9));
  EXPECT_DOUBLE_EQ(tank.energy(), 0.1);
}

// Negative withdrawal amounts are nonsensical (that's what accumulate_dissipated_power is
// for) and must be rejected rather than silently increasing the tank's energy.
TEST(EnergyTankTest, TryWithdrawRejectsNegativeDelta) {
  EnergyTank tank(5.0, 0.0, 10.0);
  EXPECT_FALSE(tank.try_withdraw(-1.0));
  EXPECT_DOUBLE_EQ(tank.energy(), 5.0);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
