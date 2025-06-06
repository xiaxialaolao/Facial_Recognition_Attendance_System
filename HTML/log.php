<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';
include 'includes/notification-functions.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 检查是否为管理员
if ($_SESSION['role'] !== 'admin') {
    header("Location: dashboard.php");
    exit();
}

// 直接获取日志记录，不检查表是否存在

// 获取当前登录用户信息
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();
$stmt->close();

// 获取最新的50条日志记录
$logs_sql = "SELECT l.log_id, l.log_level, l.message, l.source, l.created_at, u.username
             FROM system_logs l
             LEFT JOIN users u ON l.user_id = u.employee_id
             ORDER BY l.created_at DESC
             LIMIT 50";
$logs_result = $conn->query($logs_sql);
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/log-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('system_logs'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">


            <!-- 顶部标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('system_logs'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                </div>
            </div>

            <!-- 日志内容 -->
            <div class="content-header" style="margin-top: 20px;">
                <div></div> <!-- Empty div to maintain flex layout -->
                <button id="refresh-logs" class="action-button">
                    <i class="fas fa-sync-alt"></i> <?php echo __('refresh_system_info'); ?>
                </button>
            </div>

            <div class="data-table">
                <table>
                    <thead>
                        <tr>
                            <th><?php echo __('log_id'); ?></th>
                            <th><?php echo __('log_level'); ?></th>
                            <th><?php echo __('log_message'); ?></th>
                            <th><?php echo __('log_source'); ?></th>
                            <th><?php echo __('log_user'); ?></th>
                            <th><?php echo __('log_time'); ?></th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php if ($logs_result && $logs_result->num_rows > 0): ?>
                            <?php while ($log = $logs_result->fetch_assoc()): ?>
                                <tr class="log-<?php echo htmlspecialchars($log['log_level']); ?>">
                                    <td><?php echo htmlspecialchars($log['log_id']); ?></td>
                                    <td>
                                        <?php
                                            $level_class = '';
                                            $level_icon = '';

                                            switch($log['log_level']) {
                                                case 'info':
                                                    $level_class = 'text-info';
                                                    $level_icon = 'fa-info-circle';
                                                    $level_text = __('log_info');
                                                    break;
                                                case 'warning':
                                                    $level_class = 'text-warning';
                                                    $level_icon = 'fa-exclamation-triangle';
                                                    $level_text = __('log_warning');
                                                    break;
                                                case 'error':
                                                    $level_class = 'text-danger';
                                                    $level_icon = 'fa-exclamation-circle';
                                                    $level_text = __('log_error');
                                                    break;
                                                case 'debug':
                                                    $level_class = 'text-secondary';
                                                    $level_icon = 'fa-bug';
                                                    $level_text = __('log_debug');
                                                    break;
                                                default:
                                                    $level_class = 'text-info';
                                                    $level_icon = 'fa-info-circle';
                                                    $level_text = __('log_info');
                                            }
                                        ?>
                                        <span class="<?php echo $level_class; ?>" title="<?php echo __('log_'.$log['log_level'].'_description'); ?>">
                                            <i class="fas <?php echo $level_icon; ?>"></i>
                                            <?php echo $level_text; ?>
                                        </span>
                                    </td>
                                    <td class="message"><?php echo htmlspecialchars($log['message']); ?></td>
                                    <td>
                                        <?php
                                            $source = $log['source'] ?? '';
                                            $source_text = htmlspecialchars($source);
                                            $source_type = '';

                                            // 根据来源确定日志类型
                                            if (strpos($source, 'logout.php') !== false) {
                                                $source_type = 'log_type_logout';
                                            } elseif (strpos($source, 'login') !== false) {
                                                $source_type = 'log_type_login';
                                            } elseif (strpos($source, 'register') !== false) {
                                                $source_type = 'log_type_register';
                                            } elseif (strpos($source, 'addition.php') !== false) {
                                                $source_type = 'log_type_user_add';
                                            } elseif (strpos($source, 'edit.php') !== false) {
                                                $source_type = 'log_type_user_edit';
                                            } elseif (strpos($source, 'remove.php') !== false) {
                                                $source_type = 'log_type_user_delete';
                                            } elseif (strpos($source, 'user') !== false) {
                                                $source_type = 'log_type_user';
                                            } elseif (strpos($source, 'system') !== false || strpos($source, 'system_management.php') !== false) {
                                                $source_type = 'log_type_system';
                                            } elseif (strpos($source, 'attendance') !== false || strpos($source, 'statistics.php') !== false) {
                                                $source_type = 'log_type_attendance';
                                            } elseif (strpos($source, 'settings') !== false || strpos($source, 'settings.php') !== false) {
                                                $source_type = 'log_type_settings';
                                            } elseif (strpos($source, 'image_acquisition.php') !== false) {
                                                $source_type = 'log_type_image_acquisition';
                                            } elseif (strpos($source, 'script') !== false || strpos($source, '.py') !== false) {
                                                $source_type = 'log_type_script';
                                            } elseif (strpos($source, 'monitoring.php') !== false) {
                                                $source_type = 'log_type_video_stream';
                                            }

                                            if (!empty($source_type)) {
                                                echo '<span title="' . $source_text . '">' . $source_text . ' <small class="log-type-badge">' . __($source_type) . '</small></span>';
                                            } else {
                                                echo $source_text;
                                            }
                                        ?>
                                    </td>
                                    <td><?php echo htmlspecialchars($log['username'] ?? __('unknown')); ?></td>
                                    <td><?php echo date('Y-m-d H:i:s', strtotime($log['created_at'])); ?></td>
                                </tr>
                            <?php endwhile; ?>
                        <?php else: ?>
                            <tr>
                                <td colspan="6" class="text-center"><?php echo __('no_logs_found'); ?></td>
                            </tr>
                        <?php endif; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 刷新日志按钮
            document.getElementById('refresh-logs').addEventListener('click', function() {
                location.reload();
            });
        });
    </script>
</body>
</html>
