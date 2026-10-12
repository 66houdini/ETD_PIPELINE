/*
 * ETD — Electricity Theft Detection, Arduino firmware
 * Tunji (190408030) | Supervisor: Dr. O.A. Alamu
 *
 * Topology (energy balance):
 *   Source -> PZEM-A (total incoming)
 *          -> PZEM-B (metered load, Bulb 1)
 *          -> bypass branch (simulated theft, Bulb 2 via switch)
 *   GAP = Power_A - Power_B. GAP > THEFT_THRESHOLD => buzzer + LED.
 *
 * SERIAL PORT MAPPING — verified against the physical wiring:
 *   PZEM-A -> Serial1 (pins 18/19)
 *   PZEM-B -> Serial2 (pins 16/17)
 *   PZEM-C -> Serial3 (pins 14/15)
 *   Getting B and C the wrong way round makes B read the bypassed
 *   PZEM-C position (0.00 W), which forces GAP = A and latches a
 *   permanent false theft alarm.
 *
 * Output format (one line every PRINT_INTERVAL_MS, parsed by etd_hardware_integration.py):
 *   A:11.00 B:10.90 C:0.00 GAP:0.10
 */

#include <PZEM004Tv40_R4.h>
#include <LiquidCrystal_I2C.h>

PZEM004Tv40_R4 pzemA(&Serial1);   // Point A — total incoming power
PZEM004Tv40_R4 pzemB(&Serial2);   // Point B — metered load
PZEM004Tv40_R4 pzemC(&Serial3);   // Point C — currently bypassed, reads ~0.00

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int BUZZER_PIN = 8;
const int LED_PIN    = 9;
const float THEFT_THRESHOLD = 10.0;   // W; baseline noise measured at 0.0-0.3 W
const unsigned long PRINT_INTERVAL_MS = 500;

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);              // PZEM link speed — must stay 9600
  Serial2.begin(9600);
  Serial3.begin(9600);

  pzemA.begin();
  pzemB.begin();
  pzemC.begin();

  lcd.init();
  lcd.backlight();

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  // Boot marker: if this line repeats in the stream, the board is resetting.
  Serial.println("ETD_START");
}

void loop() {
  pzemA.readAll(); float pA = pzemA.getPower();
  pzemB.readAll(); float pB = pzemB.getPower();
  pzemC.readAll(); float pC = pzemC.getPower();

  // Guard against NaN from a failed sensor read, which would otherwise
  // propagate into GAP and trigger a meaningless alarm.
  if (isnan(pA)) pA = 0.0;
  if (isnan(pB)) pB = 0.0;
  if (isnan(pC)) pC = 0.0;

  float gap = pA - pB;

  Serial.print("A:");    Serial.print(pA, 2);
  Serial.print(" B:");   Serial.print(pB, 2);
  Serial.print(" C:");   Serial.print(pC, 2);
  Serial.print(" GAP:"); Serial.println(gap, 2);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("A:"); lcd.print(pA, 1); lcd.print(" B:"); lcd.print(pB, 1);
  lcd.setCursor(0, 1);
  lcd.print("C:"); lcd.print(pC, 1); lcd.print(" G:"); lcd.print(gap, 1);

  bool theft = (gap > THEFT_THRESHOLD);
  digitalWrite(LED_PIN,    theft ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, theft ? HIGH : LOW);

  delay(PRINT_INTERVAL_MS);
}
