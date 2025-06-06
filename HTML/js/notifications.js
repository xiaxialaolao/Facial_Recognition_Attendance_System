/**
 * 通知系统 - 用于在页面上显示通知消息
 * 替代传统的alert弹窗，提供更好的用户体验
 */

/**
 * 关闭通知
 * @param {HTMLElement} notification - 要关闭的通知元素
 */
function closeNotification(notification) {
    notification.classList.add('notification-hiding');
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 300);
}

// 当DOM加载完成后初始化通知区域
document.addEventListener('DOMContentLoaded', function() {
    // 如果页面上还没有通知区域，则创建一个
    if (!document.getElementById('notification-area')) {
        const notificationArea = document.createElement('div');
        notificationArea.id = 'notification-area';
        document.body.appendChild(notificationArea);
    }
});

/**
 * 显示通知
 * @param {string} message - 要显示的消息
 * @param {string} type - 通知类型 ('success', 'error', 'warning', 'info')
 * @param {number} duration - 通知显示时间（毫秒）
 */
function showNotification(message, type = 'success', duration = 3000) {
    // 获取或创建通知区域
    let notificationArea = document.getElementById('notification-area');
    if (!notificationArea) {
        notificationArea = document.createElement('div');
        notificationArea.id = 'notification-area';
        document.body.appendChild(notificationArea);
    }

    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification ${type}-notification`;

    // 设置图标
    let icon = 'fa-check-circle';
    switch (type) {
        case 'error':
            icon = 'fa-exclamation-circle';
            break;
        case 'warning':
            icon = 'fa-exclamation-triangle';
            break;
        case 'info':
            icon = 'fa-info-circle';
            break;
    }

    // 设置通知内容
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${icon}"></i>
            <div class="notification-text">
                <span class="notification-message">${message}</span>
                <span class="notification-hint">click to close</span>
            </div>
        </div>
        <button class="close-notification">
            <i class="fas fa-times"></i>
        </button>
    `;

    // 添加到通知区域
    notificationArea.appendChild(notification);

    // 添加关闭按钮事件监听器
    notification.querySelector('.close-notification').addEventListener('click', function(e) {
        e.stopPropagation(); // 阻止事件冒泡
        closeNotification(notification);
    });

    // 添加点击整个通知关闭的功能
    notification.addEventListener('click', function() {
        closeNotification(notification);
    });

    // 自动移除通知
    setTimeout(() => {
        if (notification.parentNode) {
            closeNotification(notification);
        }
    }, duration);

    // 返回通知元素，以便可以进一步操作
    return notification;
}

/**
 * 显示成功通知
 * @param {string} message - 要显示的消息
 * @param {number} duration - 通知显示时间（毫秒）
 */
function showSuccessNotification(message, duration = 3000) {
    return showNotification(message, 'success', duration);
}

/**
 * 显示错误通知
 * @param {string} message - 要显示的消息
 * @param {number} duration - 通知显示时间（毫秒）
 */
function showErrorNotification(message, duration = 3000) {
    return showNotification(message, 'error', duration);
}

/**
 * 显示警告通知
 * @param {string} message - 要显示的消息
 * @param {number} duration - 通知显示时间（毫秒）
 */
function showWarningNotification(message, duration = 3000) {
    return showNotification(message, 'warning', duration);
}

/**
 * 显示信息通知
 * @param {string} message - 要显示的消息
 * @param {number} duration - 通知显示时间（毫秒）
 */
function showInfoNotification(message, duration = 3000) {
    return showNotification(message, 'info', duration);
}
