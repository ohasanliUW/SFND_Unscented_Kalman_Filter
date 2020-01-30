#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <fstream>
#include <memory>

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // Noise covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // NIS for Radar
  double NIS_radar_;

  // NIS for Laser
  double NIS_laser_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  enum StateVariable {
      PX = 0,
      PY,
      V,
      YAW,
      YAWD,
      NU_A,
      NU_YAWDD,
  };

  // file stream objects to print NIS values to a file for analysis
  //std::shared_ptr<std::ofstream> NIS_radar_f;
  //std::shared_ptr<std::ofstream> NIS_laser_f;

  // A very bad angle normalization procedure provided in course tutorials
#define NORMALIZE_ANGLE(x)                                  \
  do {                                                      \
      while ((x) > M_PI) (x) -= 2. * M_PI;                  \
      while ((x) < -M_PI) (x) += 2. * M_PI;                 \
  } while(0)                                                \

  // High performance angle normalization based on simple mathematics
#define NORMALIZE_ANGLE_PERF(x)                             \
  do {                                                      \
    (x) = std::fmod((x) + M_PI, 2. * M_PI);                 \
    if ((x) < 0) (x) += 2. * M_PI;                          \
    (x) = (x) - M_PI;                                       \
  } while(0)                                                \

// Number of microseconds in a second
#define SECOND_IN_US 1e6

// Macro to calculate time difference between last and current measurement
#define DELTA_T(pkg, t0) (static_cast<double>((pkg).timestamp_ - (t0)) / (SECOND_IN_US))
  
 protected:
  // Method to generate augmented sigma points
  Eigen::MatrixXd AugmentedSigmaPoints();

  // Method to predict the augmented sigma points
  void SigmaPointPrediction(const Eigen::MatrixXd& Xsig_aug, double delta_t);

  // Method to predict new mean and state covariance
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> PredictMeanAndCovariance(); 

  // Method to predict radar measurement in measurement space
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> PredictRadarMeasurement(const Eigen::MatrixXd& Zsig);

  // Method to move sigma points into measurement space
  Eigen::MatrixXd SigmaPoints2MeasurementSpace();

private:
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> firstMeasurement(MeasurementPackage meas_package);
};

#endif  // UKF_H
