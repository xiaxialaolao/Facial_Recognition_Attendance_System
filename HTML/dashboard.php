<?php
// 设置缓存控制头，防止浏览器缓存
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

session_start();
include 'config.php';
include 'includes/notification-functions.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 获取当前登录用户信息
$current_user_id = $_SESSION['id'];
$current_user_sql = "SELECT username, fullname, profile_picture FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($current_user_sql);
$stmt->bind_param("i", $current_user_id);
$stmt->execute();
$current_user_result = $stmt->get_result();
$current_user = $current_user_result->fetch_assoc();

// 获取所有用户数据
$sql = "SELECT employee_id, username, fullname, role, profile_picture, created_at, updated_at FROM users";
$result = $conn->query($sql);

$users = [];
if ($result && $result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $users[] = $row;
    }
}

// 获取用户总数
$total_users = count($users);
$admin_count = 0;
$user_count = 0;

foreach ($users as $user) {
    if ($user['role'] == 'admin') {
        $admin_count++;
    } else {
        $user_count++;
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <!-- 防止浏览器缓存 -->
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <?php
    include 'includes/language-loader.php';
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('user_management'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部欢迎区 -->
            <div class="header">
                <div class="welcome-text">
                    <h1><?php echo __('welcome_back'); ?>, <?php echo htmlspecialchars($current_user['fullname']); ?></h1>
                    <p>
                        <?php
                            echo __('today_is'); ?> <?php echo date('F j, Y'); ?>
                            <?php
                                // 获取当前星期几
                                $dayOfWeek = date('l'); // 返回英文星期几名称
                                echo '(' . __($dayOfWeek) . ')'; // 使用翻译函数获取对应语言的星期几
                            ?>
                    </p>
                </div>

                <div class="header-actions">
                    <div class="search-box">
                        <i class="fas fa-search"></i>
                        <form action="search.php" method="GET">
                            <input type="text" name="search_term" placeholder="<?php echo __('search'); ?>..." aria-label="<?php echo __('search'); ?>">
                            <input type="hidden" name="search" value="1">
                        </form>
                    </div>

                    <div class="user-profile">
                        <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user['profile_picture']); ?>" alt="Profile Picture">
                    </div>
                </div>
            </div>

            <!-- 内容标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('user_management'); ?></h2>

                <button onclick="window.location.href='addition.php'" class="action-button">
                    <i class="fas fa-plus"></i>
                    <?php echo __('add_new_user'); ?>
                </button>
            </div>

            <?php if (isset($_SESSION['notification'])): ?>
                <?php
                $notification = $_SESSION['notification'];
                $message = isset($notification['message']) ? $notification['message'] : '';
                $type = isset($notification['type']) ? $notification['type'] : 'success';

                // 清除会话中的通知，防止刷新页面时再次显示
                unset($_SESSION['notification']);

                // 只有当消息不为空时才显示通知
                if (!empty($message)):
                ?>
                    <div class="<?php echo $type; ?>-message"><?php echo $message; ?></div>
                <?php endif; ?>
            <?php endif; ?>

            <!-- 数据表格区 -->
            <div class="data-table">
                <table>
                    <thead>
                        <tr>
                            <th><?php echo __('id'); ?></th>
                            <th><?php echo __('avatar'); ?></th>
                            <th><?php echo __('username'); ?></th>
                            <th><?php echo __('full_name'); ?></th>
                            <th><?php echo __('role'); ?></th>
                            <th><?php echo __('created_at'); ?></th>
                            <th><?php echo __('updated_at'); ?></th>
                            <th><?php echo __('actions'); ?></th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($users as $row): ?>
                            <tr>
                                <td><?php echo htmlspecialchars($row['employee_id']); ?></td>
                                <td>
                                    <div class="user-avatar">
                                        <img src="data:image/jpeg;base64,<?php echo base64_encode($row['profile_picture']); ?>" alt="Profile Picture">
                                    </div>
                                </td>
                                <td><?php echo htmlspecialchars($row['username']); ?></td>
                                <td><?php echo htmlspecialchars($row['fullname']); ?></td>
                                <td>
                                    <span class="status-badge <?php echo $row['role'] == 'admin' ? 'status-pending' : 'status-open'; ?>">
                                        <?php echo $row['role'] == 'admin' ? __('admin') : __('user'); ?>
                                    </span>
                                </td>
                                <td>
                                    <div class="date-time">
                                        <span class="date"><?php echo formatDate($row['created_at']); ?></span>
                                        <span class="time"><?php echo date('H:i', strtotime($row['created_at'])); ?></span>
                                    </div>
                                </td>
                                <td>
                                    <div class="date-time">
                                        <span class="date"><?php echo formatDate($row['updated_at']); ?></span>
                                        <span class="time"><?php echo date('H:i', strtotime($row['updated_at'])); ?></span>
                                    </div>
                                </td>
                                <td>
                                    <div class="action-buttons">
                                        <?php if ($_SESSION['role'] === 'admin'): ?>
                                        <!-- 管理员可以编辑和删除所有用户 -->
                                        <button onclick="window.location.href='edit.php?id=<?php echo $row['employee_id']; ?>'" class="edit-button">
                                            <i class="fas fa-edit"></i>
                                        </button>
                                        <button onclick="window.location.href='remove.php?id=<?php echo $row['employee_id']; ?>'" class="delete-button">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                        <?php else: ?>
                                        <!-- 普通用户不能编辑或删除任何用户 -->
                                        <button class="edit-button disabled" title="<?php echo __('no_permission'); ?>" disabled>
                                            <i class="fas fa-lock"></i>
                                        </button>
                                        <button class="delete-button disabled" title="<?php echo __('no_permission'); ?>" disabled>
                                            <i class="fas fa-lock"></i>
                                        </button>
                                        <?php endif; ?>
                                    </div>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- 移动端优化脚本 -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 使用会话存储来防止重复刷新
            const hasLoaded = sessionStorage.getItem('dashboardLoaded');

            if (!hasLoaded) {
                // 第一次加载页面时设置标记
                sessionStorage.setItem('dashboardLoaded', 'true');

                // 清除登录成功标记，确保按后退按钮时不会显示动画
                sessionStorage.removeItem('login_success');

                // 确保PHP会话中也清除登录成功标记
                fetch('clear_login_animation.php', {
                    method: 'POST',
                    credentials: 'same-origin'
                });
            }

            // 处理浏览器后退按钮
            window.addEventListener('popstate', function(event) {
                // 设置标记，表示是从dashboard返回的
                sessionStorage.setItem('fromDashboard', 'true');

                // 当用户点击后退按钮时，直接跳转到登录页面
                window.location.href = 'login.php';
            });

            // 设置历史状态，以便能够捕获popstate事件
            history.pushState({page: 'dashboard'}, 'Dashboard', window.location.href);

            // 检测设备类型和方向
            function checkDeviceAndOrientation() {
                const width = window.innerWidth;
                const height = window.innerHeight;
                const isLandscape = width > height;
                const isMobile = width <= 915;

                return { isMobile, isLandscape };
            }

            // 初始化表格滑动指示器
            function initTableSwipeIndicators() {
                // 获取所有表格容器
                const dataTables = document.querySelectorAll('.data-table');

                // 移除现有的指示器
                document.querySelectorAll('.swipe-indicator').forEach(indicator => {
                    indicator.remove();
                });

                dataTables.forEach(table => {
                    // 添加滑动指示器
                    const indicator = document.createElement('div');
                    indicator.className = 'swipe-indicator';

                    // 检测是否为横屏模式
                    const { isLandscape } = checkDeviceAndOrientation();

                    // 根据屏幕方向调整指示器文本
                    if (isLandscape) {
                        indicator.innerHTML = '<i class="fas fa-arrows-alt-h"></i> ← →';
                        indicator.style.cssText = 'text-align: center; padding: 3px; color: #777; font-size: 10px; background-color: #f9f9f9; border-top: 1px solid #eee; display: none;';
                    } else {
                        indicator.innerHTML = '<i class="fas fa-arrows-alt-h"></i> ← Swipe to view more →';
                        indicator.style.cssText = 'text-align: center; padding: 10px; color: #777; font-size: 12px; background-color: #f9f9f9; border-top: 1px solid #eee; display: none;';
                    }

                    // 检查表格是否需要滚动
                    if (table.scrollWidth > table.clientWidth) {
                        indicator.style.display = 'block';
                        table.appendChild(indicator);

                        // 滚动时隐藏指示器
                        table.addEventListener('scroll', function() {
                            if (this.scrollLeft > 0) {
                                indicator.style.opacity = '0.3';
                            } else {
                                indicator.style.opacity = '1';
                            }
                        });
                    }
                });
            }

            // 优化表格在横屏模式下的显示
            function optimizeTableForLandscape() {
                const { isLandscape } = checkDeviceAndOrientation();
                const tables = document.querySelectorAll('.data-table table');

                if (isLandscape) {
                    // 在横屏模式下，尝试减小表格的最小宽度，使其更适合横屏显示
                    tables.forEach(table => {
                        table.style.minWidth = '600px'; // 减小表格的最小宽度
                    });
                } else {
                    // 恢复默认值
                    tables.forEach(table => {
                        table.style.minWidth = '800px';
                    });
                }
            }

            // 初始化
            const { isMobile } = checkDeviceAndOrientation();
            if (isMobile) {
                initTableSwipeIndicators();
                optimizeTableForLandscape(); // 调用新函数优化表格
            }

            // 监听屏幕方向变化
            window.addEventListener('resize', function() {
                const { isMobile } = checkDeviceAndOrientation();
                if (isMobile) {
                    initTableSwipeIndicators();
                    optimizeTableForLandscape(); // 调用新函数优化表格
                }
            });

            // 监听屏幕方向变化事件
            window.addEventListener('orientationchange', function() {
                setTimeout(function() {
                    const { isMobile } = checkDeviceAndOrientation();
                    if (isMobile) {
                        initTableSwipeIndicators();
                        optimizeTableForLandscape(); // 调用新函数优化表格
                    }
                }, 300); // 延迟执行，确保DOM已更新
            });

            // 初始立即执行一次优化
            optimizeTableForLandscape();
        });
    </script>
</body>
</html>
