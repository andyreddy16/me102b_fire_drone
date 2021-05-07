#include <ESP32Encoder.h>
#define BIN_1 26
#define BIN_2 25

ESP32Encoder encoder;

/* encoder */
float p = 0;
float v = 0;
float p_last = 0;
const long dt = 5;

/* setting PWM properties */
const int freq = 5000;
const int ledChannel_1 = 1;
const int ledChannel_2 = 2;
const int resolution = 8;
int MAX_PWM_VOLTAGE = 255;
long i = 0;

unsigned long time_now = 0;
long t = 0;

/* PID gains */
float kp = 50;
float ki = 0.1;
float kd = 1;

float position_desired = 0;
float PWM_desired=0;
float y0_des=100;


/* feedback control */
float total_error=0;
const float total_error_max=1000;


int limit_output(int output){
  if (output>255){
    return 255;
  }
  else if (output<-255){
    return -255;
  }
  else{
    return  output;
  }
  
}

int PIDControlPosition(float position_desired) {
  float error= position_desired-p;
  total_error=total_error+error;
  cap();
  return (int) limit_output((kp*error+ki*total_error-kd*v));
}

void cap() {
  if (total_error>total_error_max){
    total_error=total_error_max;
  }
  else if (total_error<-total_error_max){
    total_error=-total_error_max;
  }
}



void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  ESP32Encoder::useInternalWeakPullResistors = UP; // Enable the weak pull up resistors
  encoder.attachHalfQuad(27, 33); // Attache pins for use as encoder pins
  encoder.setCount(0);  // set starting count value after attaching

  /* configure LED PWM functionalitites */
  ledcSetup(ledChannel_1, freq, resolution);
  ledcSetup(ledChannel_2, freq, resolution);

  /* attach the channel to the GPIO to be controlled */
  ledcAttachPin(BIN_1, ledChannel_1);
  ledcAttachPin(BIN_2, ledChannel_2);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    String reading = Serial.readStringUntil('\n');  // read from the Serial Monitor
    /* put your code here*/
    char which_gain = reading.charAt(0);
    String gain_value_string = reading.substring(1, reading.length());
    float gain_value = gain_value_string.toFloat();
    switch (which_gain) {
       case 'p':
        kp = gain_value;
        break;
       case 'i':
        ki = gain_value;
        break;
       case 'd':
        kd = gain_value;
        break;
       case 's':
        position_desired = gain_value;
        break;
       case 'w':
        PWM_desired = gain_value;
        break;
    }
  }
  /* encoder section */
  t = millis();
  p = encoder.getCount() / (75.81 * 6) * 2 * M_PI;
  v = (p - p_last) / (dt * 0.001);
  p_last = p;

  /* motor sine wave section */
  float output = PIDControlPosition(position_desired);
  if (output > 0) {
    ledcWrite(ledChannel_2, LOW);
    ledcWrite(ledChannel_1, output);
  }
  else {
    ledcWrite(ledChannel_1, LOW);
    ledcWrite(ledChannel_2, -output);
  }
  i++;
  
  Serial.println("position_desired, position");
  Serial.print(position_desired);
  Serial.print(" ");
  Serial.println(v);
  
  while (millis() < (t + dt)) {};
}

/* useful functions */
//myString.substring(from, to)
//myString.length()
//myString.toFloat()
//myString.charAt(n)
