<?php
// Start session
session_start();

// Check if user is logged in
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header('Location: login.php');
    exit;
}

// Include database connection
include 'config.php';

// Include language loader
include 'includes/language-loader.php';

// Include notification functions
include 'includes/notification-functions.php';

// Include log functions
include 'includes/log-functions.php';

// Get current user information
$current_user_id = $_SESSION['id'];
$stmt = $conn->prepare("SELECT username, fullname, profile_picture, role FROM users WHERE employee_id = ?");
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();
$stmt->close();

// Get camera stream URL
function get_camera_stream_url() {
    // 获取本机 IP 地址
    $server_ip = $_SERVER['SERVER_ADDR'];
    if ($server_ip == '::1' || $server_ip == '127.0.0.1') {
        // 如果是本地访问，尝试获取真实 IP
        $command = "hostname -I | awk '{print $1}'";
        $server_ip = trim(shell_exec($command));
    }

    // 默认端口为 5000
    $port = 5000;

    return "http://{$server_ip}:{$port}/video_feed";
}

// 获取视频流URL
$camera_stream_url = get_camera_stream_url();

// 检查视频流是否可用
function is_camera_stream_available($url) {
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_NOBODY, true);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 2); // 设置超时时间为2秒
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    return $http_code == 200;
}

// 尝试检查视频流是否可用，如果不可用则记录错误日志
$camera_available = false;
try {
    $camera_available = is_camera_stream_available($camera_stream_url);
    if (!$camera_available) {
        // 记录错误日志
        log_error("Video stream not available. Stream URL: $camera_stream_url", "monitoring.php", $_SESSION['id']);
    }
} catch (Exception $e) {
    // 如果检查过程中出现异常，也记录错误日志
    log_error("Error checking video stream: " . $e->getMessage() . ". Stream URL: $camera_stream_url", "monitoring.php", $_SESSION['id']);
}




?>

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/monitoring.css">

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('camera_monitoring'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- Main content area -->
        <div class="main-content">
            <!-- Top title area -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('camera_monitoring'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                </div>
            </div>

            <!-- Camera monitoring area -->
            <div class="monitoring-container">
                <!-- 通知区域由notifications-loader.php处理 -->

                <!-- 摄像头视频流 -->
                <div class="video-container">
                    <div class="video-stream">
                        <img src="<?php echo $camera_stream_url; ?>" alt="<?php echo __('camera_stream'); ?>" id="camera-stream">
                    </div>

                    <!-- Stream URL display -->
                    <div class="stream-info" style="margin-top: 15px; text-align: left; width: 100%;">
                        <div class="stream-url">
                            <span class="info-label"><?php echo __('stream_url'); ?>:</span>
                            <code><?php echo $camera_stream_url; ?></code>
                        </div>
                    </div>

                    <!-- Control buttons -->
                    <div style="display: flex; justify-content: flex-end; margin-top: 10px; gap: 10px;">
                        <button id="refresh-stream" style="background-color: var(--raspberry-red, #c51d4a); color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 14px; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                            <i class="fas fa-sync-alt"></i> <?php echo __('refresh_stream'); ?>
                        </button>

                        <button id="fullscreen" style="background-color: var(--raspberry-red, #c51d4a); color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 14px; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                            <i class="fas fa-expand"></i> <?php echo __('fullscreen'); ?>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 视频流控制
            const cameraStream = document.getElementById('camera-stream');
            const fullscreenBtn = document.getElementById('fullscreen');
            const refreshStreamBtn = document.getElementById('refresh-stream');

            // 添加悬停效果
            const buttons = document.querySelectorAll('#refresh-stream, #fullscreen');
            buttons.forEach(button => {
                button.addEventListener('mouseover', function() {
                    this.style.backgroundColor = '#a01540';
                    this.style.transform = 'translateY(-2px)';
                    this.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.15)';
                });

                button.addEventListener('mouseout', function() {
                    this.style.backgroundColor = 'var(--raspberry-red, #c51d4a)';
                    this.style.transform = 'translateY(0)';
                    this.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
                });
            });

            // 刷新视频流
            if (refreshStreamBtn && cameraStream) {
                refreshStreamBtn.addEventListener('click', function() {
                    // 添加时间戳参数以避免缓存
                    const timestamp = new Date().getTime();
                    const originalSrc = cameraStream.src.split('?')[0];
                    cameraStream.src = originalSrc + '?t=' + timestamp;
                });
            }

            // 全屏功能
            if (fullscreenBtn && cameraStream) {
                fullscreenBtn.addEventListener('click', function() {
                    if (!document.fullscreenElement) {
                        if (cameraStream.requestFullscreen) {
                            cameraStream.requestFullscreen();
                        } else if (cameraStream.mozRequestFullScreen) { // Firefox
                            cameraStream.mozRequestFullScreen();
                        } else if (cameraStream.webkitRequestFullscreen) { // Chrome, Safari
                            cameraStream.webkitRequestFullscreen();
                        } else if (cameraStream.msRequestFullscreen) { // IE/Edge
                            cameraStream.msRequestFullscreen();
                        }
                        fullscreenBtn.querySelector('i').className = 'fas fa-compress';
                    } else {
                        if (document.exitFullscreen) {
                            document.exitFullscreen();
                        } else if (document.mozCancelFullScreen) {
                            document.mozCancelFullScreen();
                        } else if (document.webkitExitFullscreen) {
                            document.webkitExitFullscreen();
                        } else if (document.msExitFullscreen) {
                            document.msExitFullscreen();
                        }
                        fullscreenBtn.querySelector('i').className = 'fas fa-expand';
                    }
                });

                // 监听全屏变化
                document.addEventListener('fullscreenchange', updateFullscreenButton);
                document.addEventListener('webkitfullscreenchange', updateFullscreenButton);
                document.addEventListener('mozfullscreenchange', updateFullscreenButton);
                document.addEventListener('MSFullscreenChange', updateFullscreenButton);

                function updateFullscreenButton() {
                    if (document.fullscreenElement) {
                        fullscreenBtn.querySelector('i').className = 'fas fa-compress';
                    } else {
                        fullscreenBtn.querySelector('i').className = 'fas fa-expand';
                    }
                }
            }

            // 检查视频流是否加载成功
            if (cameraStream) {
                cameraStream.onerror = function() {
                    // 视频流加载失败
                    cameraStream.style.display = 'none';
                    const videoStreamElement = document.querySelector('.video-stream');
                    if (videoStreamElement) {
                        videoStreamElement.innerHTML = `
                            <div class="video-placeholder">
                                <i class="fas fa-exclamation-triangle"></i>
                                <p><?php echo __('failed_to_load_stream'); ?></p>
                                <p><?php echo __('check_camera_running'); ?></p>
                            </div>
                        `;

                        // 错误日志已在PHP部分处理
                    }
                };
            }
        });
    </script>
</body>
</html>
