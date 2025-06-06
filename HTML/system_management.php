<?php
// Start session
session_start();

// Check if user is logged in
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header('Location: login.php');
    exit;
}

// Check if user is admin
if ($_SESSION['role'] !== 'admin') {
    header('Location: dashboard.php');
    exit;
}

// Include database connection
include 'config.php';

// Include language loader
include 'includes/language-loader.php';

// Include notification functions
include 'includes/notification-functions.php';

// Define constants
define('FRAS_ENV_PATH', '/home/xiaxialaolao/FRAS_env');
define('WEBSITE_PATH', '/home/xiaxialaolao/FRAS_env/HTML');

// Get current user information
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture, role FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();
$stmt->close();

// Handle form submission
$success_message = '';
$error_message = '';

// Function to get system information
function get_system_info() {
    $info = [];

    // CPU usage
    $cpu_command = "top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'";
    $info['cpu_usage'] = trim(shell_exec($cpu_command)) . '%';

    // CPU temperature (Raspberry Pi specific)
    $temp_command = "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null";
    $temp = shell_exec($temp_command);
    if ($temp) {
        $info['cpu_temp'] = round($temp / 1000, 1) . '°C';
    } else {
        $info['cpu_temp'] = __('unable_to_get');
    }

    // Memory usage
    $mem_command = "free -m | grep 'Mem:'";
    $mem_output = shell_exec($mem_command);
    if ($mem_output) {
        $mem_parts = preg_split('/\s+/', trim($mem_output));
        $total_mem = isset($mem_parts[1]) ? $mem_parts[1] : 0;
        $used_mem = isset($mem_parts[2]) ? $mem_parts[2] : 0;
        $info['total_mem'] = $total_mem . ' MB';
        $info['used_mem'] = $used_mem . ' MB';
        $info['mem_usage'] = ($total_mem > 0) ? round(($used_mem / $total_mem) * 100, 1) . '%' : '0%';
    } else {
        $info['total_mem'] = __('unable_to_get');
        $info['used_mem'] = __('unable_to_get');
        $info['mem_usage'] = __('unable_to_get');
    }

    // Disk usage
    $disk_command = "df -h / | grep '/'";
    $disk_output = shell_exec($disk_command);
    if ($disk_output) {
        $disk_parts = preg_split('/\s+/', trim($disk_output));
        $info['total_disk'] = isset($disk_parts[1]) ? $disk_parts[1] : __('unable_to_get');
        $info['used_disk'] = isset($disk_parts[2]) ? $disk_parts[2] : __('unable_to_get');
        $info['disk_usage'] = isset($disk_parts[4]) ? $disk_parts[4] : __('unable_to_get');
    } else {
        $info['total_disk'] = __('unable_to_get');
        $info['used_disk'] = __('unable_to_get');
        $info['disk_usage'] = __('unable_to_get');
    }

    // System load
    $load_command = "cat /proc/loadavg";
    $load_output = shell_exec($load_command);
    if ($load_output) {
        $load_parts = explode(' ', trim($load_output));
        $info['load_1min'] = isset($load_parts[0]) ? $load_parts[0] : __('unable_to_get');
        $info['load_5min'] = isset($load_parts[1]) ? $load_parts[1] : __('unable_to_get');
        $info['load_15min'] = isset($load_parts[2]) ? $load_parts[2] : __('unable_to_get');
    } else {
        $info['load_1min'] = __('unable_to_get');
        $info['load_5min'] = __('unable_to_get');
        $info['load_15min'] = __('unable_to_get');
    }

    // System uptime
    $uptime_command = "uptime -p";
    $info['uptime'] = trim(shell_exec($uptime_command));

    // Network interface information
    $network_command = "ip -o addr show | grep 'inet ' | grep -v '127.0.0.1'";
    $network_output = shell_exec($network_command);
    $info['network'] = [];
    if ($network_output) {
        $lines = explode("\n", trim($network_output));
        foreach ($lines as $line) {
            if (empty($line)) continue;
            $parts = preg_split('/\s+/', trim($line));
            $interface = isset($parts[1]) ? $parts[1] : '';
            $ip = isset($parts[3]) ? $parts[3] : '';
            if ($interface && $ip) {
                $info['network'][] = [
                    'interface' => $interface,
                    'ip' => $ip
                ];
            }
        }
    }

    // MySQL version
    if (function_exists('mysqli_get_client_info')) {
        $info['mysql_version'] = mysqli_get_client_info();
    } else {
        $mysql_version_command = "mysql --version | awk '{print $5}' | sed 's/,//'";
        $mysql_version = trim(shell_exec($mysql_version_command));
        $info['mysql_version'] = !empty($mysql_version) ? $mysql_version : __('unable_to_get');
    }

    // Python version
    $python_version_command = "python3 --version 2>&1";
    $python_version = trim(shell_exec($python_version_command));
    if (empty($python_version)) {
        $python_version_command = "python --version 2>&1";
        $python_version = trim(shell_exec($python_version_command));
    }
    $info['python_version'] = !empty($python_version) ? $python_version : __('unable_to_get');

    return $info;
}

// Handle form submission
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action'])) {
        $action = $_POST['action'];

        switch ($action) {
            case 'refresh_system_info':
                // Refresh system information
                $system_info = get_system_info();
                $success_message = __('system_info_refreshed');
                set_success_notification($success_message);
                break;

            // Process status checking has been moved to script.php

            default:
                $error_message = __('unknown_action');
                break;
        }
    }
}

// Get system information by default
$system_info = get_system_info();
?>

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/system-management-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('system_management'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- Main content area -->
        <div class="main-content">
            <!-- Top title area -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('system_management'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                </div>
            </div>

            <!-- System management area -->
            <div class="system-management-container">
                <!-- 通知区域由notifications-loader.php处理 -->

                <!-- 刷新控制 -->
                <div class="control-panel">
                    <form method="POST" class="action-form" id="system-refresh-form">
                        <input type="hidden" name="action" value="refresh_system_info">
                        <button type="submit" class="refresh-button">
                            <i class="fas fa-sync-alt"></i> <?php echo __('refresh_system_info'); ?>
                        </button>
                    </form>

                    <div class="auto-refresh-container">
                        <label class="switch">
                            <input type="checkbox" id="auto-refresh-toggle">
                            <span class="slider"></span>
                        </label>
                        <span class="refresh-text"><?php echo __('auto_refresh_every_3_minutes'); ?></span>
                    </div>

                    <div class="last-updated">
                        <?php echo __('last_updated_time'); ?>: <?php echo date('Y-m-d H:i:s'); ?>
                    </div>
                </div>

                <!-- System monitoring dashboard -->
                <div class="dashboard-cards">
                    <!-- CPU usage card -->
                    <div class="dashboard-card">
                        <i class="fas fa-microchip card-icon"></i>
                        <div class="card-title"><?php echo __('cpu_usage'); ?></div>
                        <div class="card-value"><?php echo $system_info['cpu_usage']; ?></div>
                        <div class="card-subtitle"><?php echo __('processor_usage'); ?></div>
                        <?php
                            $cpu_usage_value = floatval($system_info['cpu_usage']);
                            $cpu_class = 'progress-low';
                            if ($cpu_usage_value > 70) {
                                $cpu_class = 'progress-high';
                            } elseif ($cpu_usage_value > 40) {
                                $cpu_class = 'progress-medium';
                            }
                        ?>
                        <div class="progress-bar">
                            <div class="progress <?php echo $cpu_class; ?>" style="width: <?php echo $system_info['cpu_usage']; ?>"></div>
                        </div>
                    </div>

                    <!-- CPU temperature card -->
                    <div class="dashboard-card">
                        <i class="fas fa-thermometer-half card-icon"></i>
                        <div class="card-title"><?php echo __('cpu_temperature'); ?></div>
                        <div class="card-value"><?php echo $system_info['cpu_temp']; ?></div>
                        <div class="card-subtitle"><?php echo __('processor_temperature'); ?></div>
                        <?php
                            $cpu_temp_value = floatval($system_info['cpu_temp']);
                            $temp_class = 'progress-low';
                            $temp_percentage = 0;

                            if ($cpu_temp_value > 0) {
                                // 假设正常温度范围是0-85°C
                                $temp_percentage = min(100, ($cpu_temp_value / 85) * 100);

                                if ($cpu_temp_value > 70) {
                                    $temp_class = 'progress-high';
                                } elseif ($cpu_temp_value > 50) {
                                    $temp_class = 'progress-medium';
                                }
                            }
                        ?>
                        <div class="progress-bar">
                            <div class="progress <?php echo $temp_class; ?>" style="width: <?php echo $temp_percentage; ?>%"></div>
                        </div>
                    </div>

                    <!-- Memory usage card -->
                    <div class="dashboard-card">
                        <i class="fas fa-memory card-icon"></i>
                        <div class="card-title"><?php echo __('memory_usage'); ?></div>
                        <div class="card-value"><?php echo $system_info['mem_usage']; ?></div>
                        <div class="card-subtitle"><?php echo $system_info['used_mem']; ?> / <?php echo $system_info['total_mem']; ?></div>
                        <?php
                            $mem_usage_value = floatval($system_info['mem_usage']);
                            $mem_class = 'progress-low';
                            if ($mem_usage_value > 80) {
                                $mem_class = 'progress-high';
                            } elseif ($mem_usage_value > 60) {
                                $mem_class = 'progress-medium';
                            }
                        ?>
                        <div class="progress-bar">
                            <div class="progress <?php echo $mem_class; ?>" style="width: <?php echo $system_info['mem_usage']; ?>"></div>
                        </div>
                    </div>

                    <!-- Disk usage card -->
                    <div class="dashboard-card">
                        <i class="fas fa-hdd card-icon"></i>
                        <div class="card-title"><?php echo __('disk_usage'); ?></div>
                        <div class="card-value"><?php echo $system_info['disk_usage']; ?></div>
                        <div class="card-subtitle"><?php echo $system_info['used_disk']; ?> / <?php echo $system_info['total_disk']; ?></div>
                        <?php
                            $disk_usage_value = floatval($system_info['disk_usage']);
                            $disk_class = 'progress-low';
                            if ($disk_usage_value > 85) {
                                $disk_class = 'progress-high';
                            } elseif ($disk_usage_value > 70) {
                                $disk_class = 'progress-medium';
                            }
                        ?>
                        <div class="progress-bar">
                            <div class="progress <?php echo $disk_class; ?>" style="width: <?php echo $system_info['disk_usage']; ?>"></div>
                        </div>
                    </div>

                    <!-- System load card -->
                    <div class="dashboard-card">
                        <i class="fas fa-tachometer-alt card-icon"></i>
                        <div class="card-title"><?php echo __('system_load'); ?></div>
                        <div class="card-value"><?php echo $system_info['load_1min']; ?></div>
                        <div class="card-subtitle">
                            <?php echo __('1_minute'); ?>: <?php echo $system_info['load_1min']; ?> |
                            <?php echo __('5_minutes'); ?>: <?php echo $system_info['load_5min']; ?> |
                            <?php echo __('15_minutes'); ?>: <?php echo $system_info['load_15min']; ?>
                        </div>
                    </div>

                    <!-- System uptime card -->
                    <div class="dashboard-card">
                        <i class="fas fa-clock card-icon"></i>
                        <div class="card-title"><?php echo __('system_uptime'); ?></div>
                        <div class="card-value"><?php echo $system_info['uptime']; ?></div>
                        <div class="card-subtitle"><?php echo __('system_continuous_runtime'); ?></div>
                    </div>
                </div>

                <!-- Network information -->
                <div class="management-section">
                    <h3><i class="fas fa-network-wired"></i> <?php echo __('network_information'); ?></h3>
                    <div class="network-list">
                        <?php if (!empty($system_info['network'])): ?>
                            <?php foreach ($system_info['network'] as $network): ?>
                                <div class="network-item">
                                    <span class="network-interface"><?php echo $network['interface']; ?>:</span>
                                    <span class="network-ip"><?php echo $network['ip']; ?></span>
                                </div>
                            <?php endforeach; ?>
                        <?php else: ?>
                            <div><?php echo __('unable_to_get_network_info'); ?></div>
                        <?php endif; ?>
                    </div>
                </div>

                <!-- 添加间隔 -->
                <div style="height: 30px;"></div>

                <!-- System information -->
                <div class="management-section">
                    <h3><i class="fas fa-info-circle"></i> <?php echo __('system_info'); ?></h3>
                    <div class="system-info">
                        <div class="info-item">
                            <span class="info-label"><?php echo __('php_version'); ?>:</span>
                            <span class="info-value"><?php echo phpversion(); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('mysql_version'); ?>:</span>
                            <span class="info-value"><?php echo isset($system_info['mysql_version']) ? $system_info['mysql_version'] : __('unable_to_get'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('apache_version'); ?>:</span>
                            <span class="info-value"><?php echo function_exists('apache_get_version') ? apache_get_version() : __('unable_to_get'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('python_version'); ?>:</span>
                            <span class="info-value"><?php echo isset($system_info['python_version']) ? $system_info['python_version'] : __('unable_to_get'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('operating_system'); ?>:</span>
                            <span class="info-value"><?php echo php_uname('s') . ' ' . php_uname('r'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('max_upload_size'); ?>:</span>
                            <span class="info-value"><?php echo ini_get('upload_max_filesize'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('max_post_size'); ?>:</span>
                            <span class="info-value"><?php echo ini_get('post_max_size'); ?></span>
                        </div>
                        <div class="info-item">
                            <span class="info-label"><?php echo __('max_execution_time'); ?>:</span>
                            <span class="info-value"><?php echo ini_get('max_execution_time'); ?> <?php echo __('seconds'); ?></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 系统信息自动刷新功能
            const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
            let systemRefreshInterval;
            const THREE_MINUTES = 180000; // 3分钟 = 180000毫秒

            autoRefreshToggle.addEventListener('change', function() {
                if (this.checked) {
                    // 启动自动刷新
                    systemRefreshInterval = setInterval(function() {
                        // 使用AJAX刷新系统信息
                        refreshSystemInfo();
                    }, THREE_MINUTES); // 3分钟刷新一次

                    // 立即刷新一次
                    refreshSystemInfo();
                } else {
                    // 停止自动刷新
                    clearInterval(systemRefreshInterval);
                }
            });

            // Process status functionality has been moved to script.php

            // 系统信息刷新功能
            const systemRefreshForm = document.getElementById('system-refresh-form');
            if (systemRefreshForm) {
                systemRefreshForm.addEventListener('submit', function(e) {
                    e.preventDefault(); // 阻止表单默认提交

                    refreshSystemInfo();
                });
            }

            // 刷新系统信息函数
            function refreshSystemInfo() {
                // 显示加载状态
                const refreshButton = document.querySelector('#system-refresh-form button');
                const originalText = refreshButton.innerHTML;
                refreshButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + '<?php echo __('refreshing'); ?>...';
                refreshButton.disabled = true;

                // 创建AJAX请求
                const xhr = new XMLHttpRequest();
                xhr.open('POST', window.location.href, true);
                xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');

                xhr.onload = function() {
                    if (xhr.status === 200) {
                        // 解析返回的HTML
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(xhr.responseText, 'text/html');

                        // 更新所有仪表板卡片
                        const newCards = doc.querySelectorAll('.dashboard-card');
                        const currentCards = document.querySelectorAll('.dashboard-card');

                        if (newCards.length === currentCards.length) {
                            for (let i = 0; i < newCards.length; i++) {
                                currentCards[i].innerHTML = newCards[i].innerHTML;
                            }
                        }

                        // 更新最后更新时间
                        const lastUpdated = document.querySelector('.control-panel .last-updated');
                        if (lastUpdated) {
                            lastUpdated.textContent = '<?php echo __('last_updated_time'); ?>: ' + new Date().toLocaleString();
                        }

                        // 恢复按钮状态
                        refreshButton.innerHTML = originalText;
                        refreshButton.disabled = false;
                    }
                };

                xhr.onerror = function() {
                    // 处理错误
                    showErrorNotification('<?php echo __('error_refreshing_system_info'); ?>');
                    refreshButton.innerHTML = originalText;
                    refreshButton.disabled = false;
                };

                // 发送请求
                xhr.send('action=refresh_system_info');
            }
        });
    </script>
</body>
</html>
