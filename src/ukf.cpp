#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    // original was 30
    std_a_ = 0.2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    // original was 30
    std_yawdd_ = 0.2;

   /**
    * DO NOT MODIFY measurement noise values below.
    * These are provided by the sensor manufacturer.
    */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

   /**
    * End DO NOT MODIFY section for measurement noise values 
    */

    /**
    * TODO: Complete the initialization. See ukf.h for other member properties.
    * Hint: one or more values initialized above might be wildly off...
    */
    is_initialized_ = false;

    n_x_ = x_.rows();

    n_aug_ = n_x_ + 2;

    lambda_ = 3 - n_x_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}

void UKF::GenerateSigmaPoints() {

    // Calculate square root of covariance matrix P_
    MatrixXd sqrtP = P_.llt().matrixL();

    // First column of sigma point matrix is state vector x_
    Xsig_pred_.col(0) = x_;
      // set remaining sigma points
    for (int i = 0; i < n_x_; ++i) {
        Xsig_pred_.col(i+1)     = x_ + sqrt(lambda_ + n_x_) * sqrtP.col(i);
        Xsig_pred_.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * sqrtP.col(i);
    }
}

void UKF::AugmentedSigmaPoints(Eigen::MatrixXd& Xsig_aug_out) {
    Eigen::VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd P_aug        = MatrixXd(7, 7);

    x_aug.head(5) = x_;
    x_aug(NU_A) = 0;
    x_aug(NU_YAWDD) = 0;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;

    // Create square root matrix of P_aug
    Eigen::MatrixXd L = P_aug.llt().matrixL();

    Xsig_aug_out.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug_out.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug_out.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }
}

void UKF::SigmaPointPrediction(const Eigen::MatrixXd& Xsig_aug, Eigen::MatrixXd& Xsig_out, double delta_t) {
    // Vector to predict new X state
    auto deltaX = Eigen::VectorXd(n_x_);
    auto noise = Eigen::VectorXd(n_x_);

    for (uint8_t i = 0; i < Xsig_aug.cols(); ++i) {
        Eigen::VectorXd colm = Xsig_aug.col(i);

        auto yawd = colm(UKF::YAWD);
        auto yaw  = colm(UKF::YAW);
        auto v    = colm(UKF::V);
        // avoid division by zero
        if (std::fabs(yawd) > std::numeric_limits<double>::epsilon()) {
            deltaX(PX) = v / yawd * (std::sin(yaw + yawd * delta_t) - std::sin(yaw));
            deltaX(PY) = v / yawd * (std::cos(yaw) - std::cos(yaw + yawd * delta_t));
        } else {
            deltaX(PX) = v * delta_t * std::cos(yaw);
            deltaX(PY) = v * delta_t * std::sin(yaw);
        }

        deltaX(V)    = 0;
        deltaX(YAW)  = yawd * delta_t;
        deltaX(YAWD) = 0;

        auto nu_a = colm(UKF::NU_A);
        auto nu_yawdd = colm(UKF::NU_YAWDD);
        noise(PX)   = 0.5 * nu_a * delta_t * delta_t * std::cos(yaw);
        noise(PY)   = 0.5 * nu_a * delta_t * delta_t * std::sin(yaw);
        noise(V)    = nu_a * delta_t;
        noise(YAW)  = 0.5 * nu_yawdd * delta_t * delta_t;
        noise(YAWD) = nu_yawdd * delta_t;

        // predicted sigma point is colm + deltaX + noise;
        Xsig_pred_.col(i) = colm + deltaX + noise;
    }
}

void UKF::PredictMeanAndCovariance()
{
    // set weights
    weights_.setConstant(0.5 / (n_aug_ +  lambda_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    x_.setZero();
    P_.setZero();

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}
