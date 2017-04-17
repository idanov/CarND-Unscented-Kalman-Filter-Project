#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

auto crop = [](double x, double lower){ return fabs(x) > fabs(lower) ? x : lower; };
double lim = 1e-2;
auto angleNorm = [](double theta){
  const double rem = fmod(theta + M_PI, 2 * M_PI);
  return rem < 0 ? rem + M_PI : rem - M_PI;
};

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

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

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "Initializing UKF..." << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // get measurements from Radar
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);

      // calculate cos/sin of phi
      double cos_phi = cos(phi);
      double sin_phi = sin(phi);

      // convert from polar to cartesian coordinates
      double px = crop(rho * cos_phi, lim);
      double py = crop(-rho * sin_phi, lim);

      // initialize state
      x_ << px, py, 0, 0, 0;
      // set initial timestamp
      time_us_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize position state wiht Lidar measurements
      double px = crop(meas_package.raw_measurements_(0), lim);
      double py = crop(meas_package.raw_measurements_(1), lim);

      x_ << px, py, 0, 0, 0;
      // set initial timestamp
      time_us_ = meas_package.timestamp_;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	// dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  // make sure dt is not 0
  dt = crop(dt, lim * lim);

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*****************************************************************************
   *  Generate Sigma Points
   ****************************************************************************/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  //create noise covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_,                       0,
                     0, std_yawdd_ * std_yawdd_;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  //calculate square root of P
  MatrixXd A = ((lambda_ + n_aug_) * P_aug).llt().matrixL();
  //calculate augmented sigma points and set sigma points as columns of matrix Xsig
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.middleCols(1, n_aug_) = A.colwise() + x_aug;
  Xsig_aug.middleCols(1 + n_aug_, n_aug_) = (-A).colwise() + x_aug;

  /*****************************************************************************
   *  Predict Sigma Points
   ****************************************************************************/
  for(int i = 0; i < Xsig_aug.cols(); i++) {
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double psi = Xsig_aug(3,i);
    double psi_dot = Xsig_aug(4,i);
    double v_a = Xsig_aug(5,i);
    double v_psi = Xsig_aug(6,i);

    double px_delta = 0.0f;
    double py_delta = 0.0f;
    double v_delta = 0.0f;
    double psi_delta = psi_dot * delta_t;
    double psi_dot_delta = 0.0f;

    //avoid division by zero
    if(fabs(psi_dot) > 1e-3) {
      px_delta = (v / psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi));
      py_delta = (v / psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi));
    } else {
      px_delta = v * cos(psi) * delta_t;
      py_delta = v * sin(psi) * delta_t;
    }

    double dt2 = delta_t * delta_t;
    double px_noise = (dt2 / 2) * cos(psi) * v_a;
    double py_noise = (dt2 / 2) * sin(psi) * v_a;
    double v_noise = delta_t * v_a;
    double psi_noise = (dt2 / 2) * v_psi;
    double psi_dot_noise = delta_t * v_psi;

    //write predicted sigma points into right column
    Xsig_pred_(0, i) = px + px_delta + px_noise;
    Xsig_pred_(1, i) = py + py_delta + py_noise;
    Xsig_pred_(2, i) = v + v_delta + v_noise;
    Xsig_pred_(3, i) = psi + psi_delta + psi_noise;
    Xsig_pred_(4, i) = psi_dot + psi_dot_delta + psi_dot_noise;
  }

  /*****************************************************************************
   *  Predict Mean and Covariance
   ****************************************************************************/
  //predict state mean
  x_ = Xsig_pred_ * weights_;
  //predict state covariance matrix
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  //angle normalization
  X_diff.row(3) = X_diff.row(3).unaryExpr(angleNorm);
  P_ = X_diff * weights_.asDiagonal() * X_diff.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  /*****************************************************************************
   *  Predict Radar Measurement
   ****************************************************************************/
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++) {
      double px = Xsig_pred_(0,i);
      double py = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);

      double rho = sqrt(px*px + py*py);
      double phi = atan2(py, px);
      double rho_dot = (px * cos(psi) * v + py * sin(psi) * v) / rho;

      Zsig(0,i) = rho;
      Zsig(1,i) = phi;
      Zsig(2,i) = rho_dot;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //calculate mean predicted measurement
  z_pred = Zsig * weights_;
  //calculate measurement covariance matrix S
  MatrixXd Z_diff = Zsig.colwise() - z_pred;
  //angle normalization
  Z_diff.row(1) = Z_diff.row(1).unaryExpr(angleNorm);
  //measurement noise matrix
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_ * std_radr_,                         0,                       0,
                           0, std_radphi_ * std_radphi_,                       0,
                           0,                         0, std_radrd_ * std_radrd_;

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S = Z_diff * weights_.asDiagonal() * Z_diff.transpose() + R;
  /*****************************************************************************
   *  Update state and covariance
   ****************************************************************************/
  //incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  //calculate cross correlation matrix
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  //angle normalization
  X_diff.row(3) = X_diff.row(3).unaryExpr(angleNorm);
  Tc = X_diff * weights_.asDiagonal() * Z_diff.transpose();
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  z_diff(1) = angleNorm(z_diff(1));
  x_ += K * z_diff;
  P_ -= K * S * K.transpose(); 
}
