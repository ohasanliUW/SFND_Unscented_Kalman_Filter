#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

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
    std_yawdd_ = 0.3;

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

    n_x_ = 5;

    n_aug_ = 7;

    lambda_ = 3 - n_aug_;

    NIS_radar_ = NAN;
    NIS_laser_ = NAN;

    // Weights are always the same for given lambda_
    weights_ = Eigen::VectorXd(2 * n_aug_ + 1);
    weights_.setConstant(0.5 / (n_aug_ +  lambda_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    Q_ = Eigen::MatrixXd::Zero(2,2);
    Q_(0, 0) = std_a_ * std_a_;
    Q_(1, 1) = std_yawdd_ * std_yawdd_;

    H_ = Eigen::MatrixXd::Zero(2, 5);
    H_.topLeftCorner(2, 2) = Eigen::MatrixXd::Identity(2, 2);
    std::cout << H_ << std::endl;

    R_ = Eigen::MatrixXd(2, 2);
    R_ << std_laspx_ * std_laspx_, 0,
          0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

static std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
firstMeasurement(MeasurementPackage meas_package, uint8_t n_x)
{
    Eigen::VectorXd x = Eigen::VectorXd(n_x);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n_x, n_x);
    P(0, 0) = 0.15;
    P(1, 1) = 0.15;

    if (MeasurementPackage::LASER == meas_package.sensor_type_) {
        x << meas_package.raw_measurements_(0), // px
             meas_package.raw_measurements_(1), // py
             0, 0, 0; // v, yaw, yawd
    } else {
        assert(MeasurementPackage::RADAR == meas_package.sensor_type_);

        double r     = meas_package.raw_measurements_(0);
        double phi   = meas_package.raw_measurements_(1);
        double r_dot = meas_package.raw_measurements_(2);

        double px    = r * std::cos(phi);
        double py    = r * std::sin(phi);
        double vx    = r_dot * std::cos(phi);
        double vy    = r_dot * std::sin(phi);
        double v     = std::sqrt(vx * vx + vy * vy);

        x << px, py, v, 0, 0;
    }

    return std::make_tuple(x, P);
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
    if (!is_initialized_) {
        // First measurement
        std::tie(x_, P_) = firstMeasurement(meas_package, n_x_);
        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;
        return;
    }

    // Prediction step same for both Lidar and Radar measurements
    Prediction(DELTA_T(meas_package, time_us_));

    if (MeasurementPackage::LASER == meas_package.sensor_type_ && use_laser_) {
        UpdateLidar(meas_package);
    } else if (MeasurementPackage::RADAR == meas_package.sensor_type_ && use_radar_) {
        UpdateRadar(meas_package);
    }

    time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    // First, generate sigma points
    Eigen::MatrixXd Xsig_aug = AugmentedSigmaPoints();

    // Use augemnted sigma points to predict sigma points
    SigmaPointPrediction(Xsig_aug, delta_t);

    // Use results of sigma point prediction to compute mean and covariance
    std::tie(x_, P_) = PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

    Eigen::VectorXd z_pred = H_ * x_;
    Eigen::VectorXd y      = meas_package.raw_measurements_ - z_pred;
    Eigen::MatrixXd Ht     = H_.transpose();
    Eigen::MatrixXd S      = H_ * P_ * Ht + R_;
    Eigen::MatrixXd Si     = S.inverse();
    Eigen::MatrixXd K      = P_ * Ht * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    Eigen::MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;

    NIS_laser_ = y.transpose() * Si * y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
    auto Zsig        = SigmaPoints2MeasurementSpace();
    auto [z_pred, S] = PredictRadarMeasurement(Zsig);
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, z_pred.rows());

    for (int i = 0; i < Xsig_pred_.cols(); ++i) {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        NORMALIZE_ANGLE_PERF(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NORMALIZE_ANGLE_PERF(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    NORMALIZE_ANGLE_PERF(z_diff(1));

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    // Compute NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    //std::cerr << NIS_radar_ << std::endl;
}

Eigen::MatrixXd
UKF::AugmentedSigmaPoints() {
    Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Eigen::VectorXd x_aug    = Eigen::VectorXd::Zero(n_aug_);
    Eigen::MatrixXd P_aug    = Eigen::MatrixXd::Zero(n_aug_, n_aug_);

    x_aug.head(n_x_) = x_;

    P_aug.topLeftCorner(5,5) = P_;
    P_aug.bottomRightCorner(2, 2) = Q_;

    // Create square root matrix of P_aug
    Eigen::MatrixXd L = P_aug.llt().matrixL();

    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    return Xsig_aug;
}

void UKF::SigmaPointPrediction(const Eigen::MatrixXd& Xsig_aug, double delta_t) {
    // Vector to predict new X state
    auto deltaX = Eigen::VectorXd(n_x_);
    auto noise = Eigen::VectorXd(n_x_);
    Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

    for (uint8_t i = 0; i < Xsig_pred_.cols(); ++i) {
        Eigen::VectorXd colm = Xsig_aug.col(i);

        double yawd = colm(UKF::YAWD);
        double yaw  = colm(UKF::YAW);
        double v    = colm(UKF::V);
        // avoid division by zero
        if (std::fabs(yawd) > std::numeric_limits<double>::epsilon()) {
            deltaX(PX) = (v / yawd) * (std::sin(yaw + yawd * delta_t) - std::sin(yaw));
            deltaX(PY) = (v / yawd) * (std::cos(yaw) - std::cos(yaw + yawd * delta_t));
        } else {
            deltaX(PX) = v * delta_t * std::cos(yaw);
            deltaX(PY) = v * delta_t * std::sin(yaw);
        }

        deltaX(V)    = 0;
        deltaX(YAW)  = yawd * delta_t;
        deltaX(YAWD) = 0;

        double nu_a = colm(UKF::NU_A);
        double nu_yawdd = colm(UKF::NU_YAWDD);
        noise(PX)   = 0.5 * nu_a * delta_t * delta_t * std::cos(yaw);
        noise(PY)   = 0.5 * nu_a * delta_t * delta_t * std::sin(yaw);
        noise(V)    = nu_a * delta_t;
        noise(YAW)  = 0.5 * nu_yawdd * delta_t * delta_t;
        noise(YAWD) = nu_yawdd * delta_t;

        // predicted sigma point is colm + deltaX + noise;
        Xsig_pred_.col(i) = colm.topRows(5) + deltaX + noise;
    }

}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
UKF::PredictMeanAndCovariance()
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n_x_);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n_x_, n_x_);

    // set weights

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        VectorXd x_diff = Xsig_pred_.col(i) - x;

        // angle normalization
        NORMALIZE_ANGLE_PERF(x_diff(3));

        P += weights_(i) * x_diff * x_diff.transpose();
    }

    return std::make_tuple(x, P);
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
UKF::PredictRadarMeasurement(const Eigen::MatrixXd& Zsig)
{
    // Radar measurement dimension is 3 (r, phi, r_dot)
    uint8_t n_z = 3;

    // mean predicted measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);

    // measurement covariance matrix in measurement space
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);

    // mean predict measurement
    for (int i = 0; i < Zsig.cols(); ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < Zsig.cols(); ++i) {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // normalize angle
        NORMALIZE_ANGLE_PERF(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_ , 0,
        0, 0,std_radrd_ * std_radrd_;
    S = S + R;

    return std::make_tuple(z_pred, S);
}

Eigen::MatrixXd
UKF::SigmaPoints2MeasurementSpace()
{
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(3, 2 * n_aug_ + 1);
    // transform sigma points into measurement space
    for (int i = 0; i < Xsig_pred_.cols(); ++i) {
        double p_x = Xsig_pred_(UKF::PX, i);
        double p_y = Xsig_pred_(UKF::PY, i);
        double v   = Xsig_pred_(UKF::V, i);
        double yaw = Xsig_pred_(UKF::YAW, i);

        // range r
        double r = std::sqrt(p_x * p_x + p_y * p_y);

        // angle phi
        double phi = std::atan2(p_y, p_x);

        // r_dot
        double r_dot = (p_x * std::cos(yaw) * v + p_y * std::sin(yaw) * v) / r;

        Zsig(0, i) = r; Zsig(1, i) = phi; Zsig(2, i) = r_dot;
    }

    return Zsig;
}
