/*
 * ============================================================================
 * SERVO CONTROLLER FOR PHYSICAL RESERVOIR COMPUTING
 * ============================================================================
 * Morphological Computation Group
 * University of Bristol
 * 2025
 * 
 * Description:
 *   Arduino code to drive a servo motor with a three-frequency sinusoidal
 *   signal for physical reservoir computing experiments with crumpled paper.
 *   Generates composite signal: sin(2πf₁t) × sin(2πf₂t) × sin(2πf₃t)
 * 
 * Hardware:
 *   - Arduino Uno/Nano
 *   - Servo motor connected to pin 3
 *   - Serial communication at 115200 baud
 * Version: 1.0
 * ============================================================================
 */

#include <Servo.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================
const int SERVO_PIN = 3;              // PWM pin for servo control
const int SERVO_MIN_US = 1000;        // Minimum pulse width (microseconds)
const int SERVO_MAX_US = 2000;        // Maximum pulse width (microseconds)
const int SERVO_CENTER_US = 1500;     // Center/neutral position

// ============================================================================
// TIMING PARAMETERS
// ============================================================================
const unsigned long UPDATE_INTERVAL_MS = 20;  // Update servo every 20ms (50Hz)

// ============================================================================
// SIGNAL PARAMETERS
// ============================================================================
// Base frequencies for the three-frequency composite signal (Hz)
const float FREQ_1 = 2.11;
const float FREQ_2 = 3.73;  
const float FREQ_3 = 4.33;

// User-adjustable parameters
float frequencyMultiplier = 1.0;      // Scale factor for all frequencies
float amplitudeScale = 0.5;           // Output amplitude (0.0 to 1.0)

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================
Servo servo;                          // Servo object
unsigned long startTime;              // Time reference for signal generation
unsigned long lastUpdateTime;         // Last servo update timestamp

// ============================================================================
// SIGNAL GENERATION
// ============================================================================

/**
 * Generate the three-frequency composite signal
 * @param timeSeconds - Time in seconds since start
 * @return Signal value between -1.0 and 1.0
 */
float generateCompositeSignal(float timeSeconds) {
  // Calculate each frequency component
  float component1 = sin(TWO_PI * FREQ_1 * frequencyMultiplier * timeSeconds);
  float component2 = sin(TWO_PI * FREQ_2 * frequencyMultiplier * timeSeconds);
  float component3 = sin(TWO_PI * FREQ_3 * frequencyMultiplier * timeSeconds);
  
  // Multiply components to create composite signal
  float composite = component1 * component2 * component3;
  
  // Scale by amplitude factor
  return composite * amplitudeScale;
}

/**
 * Convert signal value to servo pulse width
 * @param signal - Signal value between -1.0 and 1.0
 * @return Pulse width in microseconds
 */
int signalToPulseWidth(float signal) {
  // Clamp signal to valid range
  signal = constrain(signal, -1.0, 1.0);
  
  // Map from [-1, 1] to [SERVO_MIN_US, SERVO_MAX_US]
  // signal = -1 → SERVO_MIN_US
  // signal =  0 → SERVO_CENTER_US  
  // signal =  1 → SERVO_MAX_US
  int pulseWidth = SERVO_CENTER_US + (int)(signal * (SERVO_MAX_US - SERVO_MIN_US) / 2.0);
  
  return constrain(pulseWidth, SERVO_MIN_US, SERVO_MAX_US);
}

// ============================================================================
// SERIAL COMMUNICATION
// ============================================================================

/**
 * Display current settings
 */
void printSettings() {
  Serial.println(F("=== Current Settings ==="));
  Serial.print(F("Frequency Multiplier: "));
  Serial.println(frequencyMultiplier, 3);
  Serial.print(F("Amplitude Scale: "));
  Serial.println(amplitudeScale, 3);
  Serial.print(F("Base Frequencies (Hz): "));
  Serial.print(FREQ_1, 2); Serial.print(F(", "));
  Serial.print(FREQ_2, 2); Serial.print(F(", "));
  Serial.println(FREQ_3, 2);
  Serial.println(F("========================"));
}

/**
 * Display help menu
 */
void printHelp() {
  Serial.println(F("\n=== Servo Controller Commands ==="));
  Serial.println(F("f <value>  : Set frequency multiplier (e.g., 'f 0.5')"));
  Serial.println(F("a <value>  : Set amplitude scale 0-1 (e.g., 'a 0.8')"));
  Serial.println(F("r          : Reset to default values"));
  Serial.println(F("s          : Show current settings"));
  Serial.println(F("h          : Show this help menu"));
  Serial.println(F("=================================\n"));
}

/**
 * Process serial commands
 */
void processSerialCommand() {
  if (!Serial.available()) return;
  
  char command = Serial.read();
  
  switch (command) {
    case 'f':  // Set frequency multiplier
    case 'F':
      {
        float value = Serial.parseFloat();
        if (value > 0 && value <= 10.0) {
          frequencyMultiplier = value;
          Serial.print(F("Frequency multiplier set to: "));
          Serial.println(frequencyMultiplier, 3);
        } else {
          Serial.println(F("Error: Frequency multiplier must be between 0 and 10"));
        }
      }
      break;
      
    case 'a':  // Set amplitude scale
    case 'A':
      {
        float value = Serial.parseFloat();
        if (value >= 0 && value <= 1.0) {
          amplitudeScale = value;
          Serial.print(F("Amplitude scale set to: "));
          Serial.println(amplitudeScale, 3);
        } else {
          Serial.println(F("Error: Amplitude must be between 0 and 1"));
        }
      }
      break;
      
    case 'r':  // Reset to defaults
    case 'R':
      frequencyMultiplier = 1.0;
      amplitudeScale = 0.5;
      Serial.println(F("Reset to default values"));
      printSettings();
      break;
      
    case 's':  // Show settings
    case 'S':
      printSettings();
      break;
      
    case 'h':  // Help
    case 'H':
    case '?':
      printHelp();
      break;
      
    case '\n':  // Ignore newlines
    case '\r':
    case ' ':
      break;
      
    default:
      Serial.print(F("Unknown command: '"));
      Serial.print(command);
      Serial.println(F("'. Type 'h' for help"));
  }
  
  // Clear any remaining characters in buffer
  while (Serial.available()) {
    Serial.read();
  }
}

// ============================================================================
// ARDUINO MAIN FUNCTIONS
// ============================================================================

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println(F("\n================================"));
  Serial.println(F("Physical Reservoir Computing"));
  Serial.println(F("Servo Controller v1.0"));
  Serial.println(F("University of Bristol, 2025"));
  Serial.println(F("================================\n"));
  
  // Initialize servo
  servo.attach(SERVO_PIN);
  servo.writeMicroseconds(SERVO_CENTER_US);  // Start at center position
  
  // Initialize timing
  startTime = millis();
  lastUpdateTime = startTime;
  
  // Display initial settings
  printSettings();
  Serial.println(F("Type 'h' for help\n"));
}

void loop() {
  // Process any serial commands
  processSerialCommand();
  
  // Check if it's time to update servo
  unsigned long currentTime = millis();
  if (currentTime - lastUpdateTime >= UPDATE_INTERVAL_MS) {
    lastUpdateTime = currentTime;
    
    // Calculate time in seconds since start
    float elapsedSeconds = (currentTime - startTime) / 1000.0;
    
    // Generate composite signal
    float signal = generateCompositeSignal(elapsedSeconds);
    
    // Convert to servo pulse width and update servo
    int pulseWidth = signalToPulseWidth(signal);
    servo.writeMicroseconds(pulseWidth);
    
    // Optional: Print current values for debugging (uncomment if needed)
    // Serial.print("t="); Serial.print(elapsedSeconds, 2);
    // Serial.print(" signal="); Serial.print(signal, 3);
    // Serial.print(" pulse="); Serial.println(pulseWidth);
  }
}