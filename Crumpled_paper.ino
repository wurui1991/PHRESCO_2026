/*
   Three-frequency driver for EMAX ES9256 II HV
   • ampScale 0…1 (1 = full 1500→1900 µs throw)
   • freqMult >0
   • waveform: 0 sine | 1 square | 2 random
   • randPeriod: period for random wave updates (ms)
*/

#include <Servo.h>

#define CTRL_PIN        3
const int MIN_US = 800;        // neutral / start of travel
const int MAX_US = 2200;        // full positive travel
const unsigned long STEP_MS = 5;
const int SLEW_US = 500;

// base frequencies [Hz]
const float f1 = 2.11, f2 = 3.73, f3 = 4.33;

Servo servo;
unsigned long t0, tLast;
float  freqMult = 0.5;
float  ampScale = 1;          // 0 … 1
int    pulse    = MIN_US;

// Random wave control
unsigned long randPeriod = 500;    // ms between random value updates
unsigned long tLastRandom = 0;     // last time random value was updated
float currentRandomValue = 0;      // current random value (-1 to 1)

enum Wave { W_SINE, W_SQUARE, W_RANDOM } mode = W_SINE;

// ---------- helpers ----------
float squareWave(float freq, float tSec) {
  return (fmodf(tSec * freq, 1.0f) < 0.5f) ?  1.0f : -1.0f;
}

float getRandomWave(unsigned long now) {
  // Check if it's time to update the random value
  if (now - tLastRandom >= randPeriod) {
    tLastRandom = now;
    currentRandomValue = random(-1000, 1000) / 1000.0f;  // −1 … +1
  }
  return currentRandomValue;
}

float composite(float tSec, unsigned long now) {
  switch (mode) {
    case W_SQUARE:
      return squareWave(f1 * freqMult, tSec) *
             squareWave(f2 * freqMult, tSec) *
             squareWave(f3 * freqMult, tSec);
    case W_RANDOM:
      return getRandomWave(now);
    default: // W_SINE
      return sinf(TWO_PI * f1 * freqMult * tSec) *
             sinf(TWO_PI * f2 * freqMult * tSec) *
             sinf(TWO_PI * f3 * freqMult * tSec);
  }
}

/* ---- unipolar mapping for ES9256 (0…1 → 1500…1900 µs) ---- */
inline int sigToPulse(float sig /* −1…+1 */) {
  float uni = (sig * ampScale + 1.0f) * 0.5f;        // 0…1
  return MIN_US + (int)(uni * (MAX_US - MIN_US));
}

// ---------- serial control ----------
void parseSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil(' ');
  cmd.trim();

  if (cmd == "m") {
    float k = Serial.parseFloat();
    if (k > 0) freqMult = k;
  }
  else if (cmd == "a") {
    float k = Serial.parseFloat();
    ampScale = constrain(k, 0.0f, 1.0f);
  }
  else if (cmd == "w") {
    int w = Serial.parseInt();
    if (w >= 0 && w <= 2) mode = (Wave)w;
  }
  else if (cmd == "r") {
    unsigned long p = Serial.parseInt();
    if (p >= 10) randPeriod = p;  // minimum 10ms
  }
  else if (cmd == "?") {
    Serial.print(F("freqMult=")); Serial.print(freqMult, 3);
    Serial.print(F("  ampScale=")); Serial.print(ampScale, 3);
    Serial.print(F("  waveform=")); Serial.print((int)mode);
    Serial.print(F("  randPeriod=")); Serial.print(randPeriod);
    Serial.println(F(" ms"));
  }
  else if (cmd == "h" || cmd == "help") {
    Serial.println(F("Commands:"));
    Serial.println(F("  m <freq>   - Set frequency multiplier (>0)"));
    Serial.println(F("  a <scale>  - Set amplitude scale (0-1)"));
    Serial.println(F("  w <mode>   - Set waveform (0=sine, 1=square, 2=random)"));
    Serial.println(F("  r <ms>     - Set random wave period (≥10 ms)"));
    Serial.println(F("  ?          - Show current settings"));
    Serial.println(F("  h/help     - Show this help"));
  }
}

// ---------- Arduino life-cycle ----------
void setup() {
  servo.attach(CTRL_PIN);
  Serial.begin(115200);
  randomSeed(analogRead(A0));
  t0 = millis();
  tLastRandom = t0;
  
  Serial.println(F("Servo Controller Ready. Type 'help' for commands."));
}

void loop() {
  parseSerial();

  unsigned long now = millis();
  if (now - tLast < STEP_MS) return;
  tLast = now;

  float sig  = composite((now - t0) / 1000.0f, now);   // −1…+1
  int   tgt  = sigToPulse(sig);

  // slew-rate limiter
  pulse += constrain(tgt - pulse, -SLEW_US, SLEW_US);
  servo.writeMicroseconds(pulse);
}